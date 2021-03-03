
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
print(tf.__version__)
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, \
    Conv1D, BatchNormalization, MaxPooling1D, UpSampling1D, LSTM,Add
from tensorflow.python.keras.layers import Cropping1D
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
from tensorflow.keras.preprocessing import sequence
# I'm not sure of this import, most people import "Layer" just from Keras
# from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils.np_utils import to_categorical

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utilss import *
_model_name = 'lstm_model'


def autoencoder_model(timesteps, input_dim):
    inputs = Input(shape=(timesteps, input_dim), name='input')  # (,7813,6)
    activation = 'sigmoid'  # sigmoid
    activation_last = 'sigmoid'  # relu

    maxpoolsize = 2
    latent_dim = input_dim  # 没有意义，和xyz轴已经不对应了
    kernelsize = 8

    x = Conv1D(16, kernelsize, activation=activation, padding='same', use_bias=True)(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(maxpoolsize, padding='same')(x)
    x = Conv1D(8, kernelsize, activation=activation, padding='same', use_bias=True,
               input_shape=(timesteps, input_dim))(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(maxpoolsize, padding='same')(x)
    x = Conv1D(latent_dim, kernelsize, activation=activation_last, padding='same', use_bias=True,
               input_shape=(timesteps, input_dim))(x)
    x = BatchNormalization(axis=-1)(x)
    encoded = MaxPooling1D(2, padding='same')(x)  # (,977,2)

    x = Conv1D(latent_dim, kernelsize, activation=activation, padding='same', use_bias=True,
               input_shape=(timesteps, input_dim))(encoded)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling1D(maxpoolsize)(x)
    x = Conv1D(8, kernelsize, activation=activation, padding='same', use_bias=True,
               input_shape=(timesteps, input_dim))(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling1D(maxpoolsize)(x)
    x = Conv1D(16, kernelsize, activation=activation, padding='same', use_bias=True,
               input_shape=(timesteps, input_dim))(
        x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling1D(maxpoolsize)(x)
    # print('AE timesteps', timesteps)
    # print('AE x.shape', x.shape)
    n_crop = int(x.shape[1] - timesteps)
    x = Cropping1D(cropping=(0, n_crop))(x)

    decoded = Conv1D(input_dim, kernelsize, activation='linear', padding='same', use_bias=False,
                     input_shape=(timesteps, input_dim), name='autoencoderl')(x)

    autoencoder = Model(inputs, decoded)
    # autoencoder.compile(optimizer='Adam', loss='mse')  # mine
    # autoencoder.compile(optimizer='rmsprop', loss='mse')
    # autoencoder.summary()
    encoder = Model(inputs, encoded, name='encoded_layer')
    return autoencoder, encoder


def domain_model(encoder):
    lambdal=0.999
    flip_layer = GradientReversalLayer()
    dann_in = flip_layer(encoder.output)
    # dann_in = Conv1D(6, 5, activation="sigmoid", padding='same', use_bias=True)(dann_in)
    domain_classifier = Flatten(name="do4")(dann_in)
    domain_classifier = BatchNormalization(name="do5")(domain_classifier)
    domain_classifier = Dense(64, activation='softmax', name="do6")(domain_classifier)
    domain_classifier = Dropout(0.5)(domain_classifier)
    domain_classifier = Dense(16, activation='softmax', name="do7")(domain_classifier)
    domain_classifier = Activation("relu", name="do8")(domain_classifier)
    dann_out = Dense(2, activation='softmax', name="domain")(domain_classifier)
    domain_classification_model = Model(inputs=encoder.input, outputs=dann_out)
    return domain_classification_model

@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return dy
        # return 1.0 * tf.negative(dy)
    return tf.identity(x), grad

class GradientReversalLayer(Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs):
        return GradientReversalOperator(inputs)
        # return GradientReversalOperator(inputs, self.lambdal)

class ATTMODEL:
    def __init__(self, timesteps, input_dim):
        super(ATTMODEL, self).__init__()

        self.timesteps = timesteps
        self.input_dim = input_dim
        self.autoencoder = self.encoder = self.domain_classification_model = self.comb_model = None


    def initialize(self):
        # first: autoencoder model
        self.autoencoder, self.encoder = autoencoder_model(self.timesteps, self.input_dim)

        # domain adversarial learning: ae+domain
        self.domain_classification_model = domain_model(self.encoder)
        self.domain_classification_model.compile(optimizer="Adam",
                                                 loss=['binary_crossentropy'], metrics=['accuracy'])

        self.autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        # self.autoencoder.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

        self.comb_model = Model(inputs=self.autoencoder.input,
                                outputs=[self.autoencoder.output, self.domain_classification_model.output])
        self.comb_model.compile(optimizer="Adam",
                                loss=['mse', 'binary_crossentropy'], loss_weights=[1, 1], metrics=['accuracy'], )
                                # loss=['mse', 'binary_crossentropy'], loss_weights=[1, 15], metrics=['accuracy'], )
        self.comb_model.summary()
        print('Finished initializing model structure......')
        return


# load data of two folders respectively
datasetrootdir = r'.\dataset'
normal_dir_name = os.path.join(datasetrootdir, 'normal')
mutant_dir_name = os.path.join(datasetrootdir, 'mutant')
normal_data = get_orig_data(normal_dir_name)
mutant_data = get_orig_data(mutant_dir_name)
# normal_data, mutant_data = normalize_list(normal_data, mutant_data, bias=0.1)
maxlen = get_max_length(normal_data, mutant_data)

# transform the list to same sequence length
X_normal_train = sequence.pad_sequences(normal_data, maxlen=maxlen, dtype='float64', padding='post',
                                        truncating='post', value=-1.0)
X_mutant_train = sequence.pad_sequences(mutant_data, maxlen=maxlen, dtype='float64', padding='post',
                                            truncating='post', value=-1.0)
Y_normal = hotvec(1, 0, len(normal_data)).reshape([-1, 1])
Y_mutant = hotvec(1, 1, len(mutant_data)).reshape([-1, 1])

X_train = np.concatenate((X_normal_train, X_mutant_train))
Y_train = np.concatenate((Y_normal, Y_mutant))
y_adversarial = to_categorical(Y_train)
num_steps = 5000

input_dim = np.array(X_train).shape[-1]
# input_dim = 2
# dtc = ATTMODEL(timesteps=0, input_dim=input_dim)
dtc = ATTMODEL(timesteps=maxlen, input_dim=input_dim)
dtc.initialize()
results = []
# loss = tf.keras.losses.SparseCategoricalCrossentropy()
for epoch in range(num_steps):
    print('run epoch:' + str(epoch))

    # dtc.comb_model.fit(X_train, [yb, y_adversarial], callbacks=[LearningRateReducerCb()], epochs=1)
    stats = dtc.comb_model.train_on_batch(X_train, [X_train, y_adversarial])
    stats_test = dtc.comb_model.test_on_batch(X_train, [X_train, y_adversarial])
    # stats = dtc.domain_classification_model.train_on_batch(X_train,  y_adversarial)
    # dtc.domain_classification_model.layers[0].trainable=False
    print(stats)
    results.append(stats)
aa = np.array(results)
plt.plot(aa[:,1])
plt.title('ae loss')
plt.savefig('aeloss.png')
plt.show()


pca = PCA(n_components=2)
emb_all = dtc.encoder.predict(X_normal_train)
pca_emb = pca.fit_transform(emb_all)
emb_all2 = dtc.encoder.predict(X_mutant_train)
pca_emb2 = pca.fit_transform(emb_all2)
plt.scatter(pca_emb[:, 0], pca_emb[:, 1], c=Y_normal, cmap='coolwarm', alpha=0.4)
plt.scatter(pca_emb2[:, 0], pca_emb2[:, 1], c=Y_mutant, cmap='cool', alpha=0.4)



aa = np.array(results)

plt.figure(2)
plt.plot(aa[:,3],'r')
plt.plot(aa[:,4],'b')
plt.show()
plt.close()
