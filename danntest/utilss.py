import numpy as np
import pandas as pd
import re
import os
import glob
from sklearn.datasets import make_moons, make_blobs
from sklearn.decomposition import PCA

import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf


_model_name = 'lstm_model'

def get_filelist(dirname, savebinary=False):
    if savebinary:
        filelist = glob.glob(dirname + '/*.npy')
    else:
        filelist = glob.glob(dirname + '/*.csv')
    filelist.sort(key=cmp_to_key(compare_filename))
    return np.array(filelist)

def filename_from_fullpath(path, without_extension=False):
    filename = os.path.basename(path)
    filename_number = '-1'
    if without_extension:
        filename, ext = os.path.splitext(filename)
        filename_number = re.findall(r"\d+", filename)[0]
    # return filename
    return filename_number

def compare_filename(file1, file2):
    f1 = filename_from_fullpath(file1, True)
    f2 = filename_from_fullpath(file2, True)
    return int(f1) - int(f2)

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def get_orig_data(dirname, include_time=False):
    filelist = get_filelist(dirname)
    data, filenames= [], []
    for filepath in filelist:
        # tmp = np.genfromtxt(filepath, delimiter=',', skip_header=1) #
        tmp = pd.read_csv(filepath, delimiter=',').values
        acc_data = tmp[:, 1:]
        # filename = filepath.split('\\')[-1][:-4]
        # filenames.append(filename)
        if (not include_time) :
            arr = np.delete(acc_data, 0, axis=0)
            data.append(arr)
        else:
            data.append(acc_data)
    return np.array(data)

def get_max_length(normal_data, mutant_data):
    length = np.array([])
    for data in normal_data:
        length = np.append(length, len(data))
    for data in mutant_data:
        length = np.append(length, len(data))

    return int(np.max(length))

def hotvec(dim, label, num):
    ret = []
    for i in range(num):
        vec = [0] * dim
        vec[0] = label
        # vec[label] = 1
        ret.append(vec)
    return np.array(ret)

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

    return

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    # new_lr = old_lr * 0.99
    p = float(epoch) / num_steps
    l = 2. / (1. + np.exp(-10. * p)) - 1
    new_lr = 0.01 / (1. + 10 * p) ** 0.75
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

class GradientCallback(tf.keras.callbacks.Callback):
    console = True
    def on_epoch_end(self, epoch, logs=None):
        # x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            # t.watch(x_tensor)
            # loss = self.model(x_tensor)
            predictions = self.model.predict(input, training=False)
            lossvalue = loss(yb,predictions)
            # return t.gradient(loss, x_tensor).numpy()
        print(tape.gradient(lossvalue, self.model.trainable_variables))

def standardization(input_array, mean, std, bias=0.0):
    return ((input_array - mean) / np.maximum(std, 10 ** -5)) + bias

def normalize_list(normal_list, mutant_list, bias=0.0):
    nc = np.concatenate(normal_list)
    mc = np.concatenate(mutant_list)
    con = np.concatenate((nc, mc)) # 拼接

    mean = np.mean(con, axis=0)
    std = np.std(con, dtype=float, axis=0)

    ret_normal = []
    ret_mutant = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], mean, std, bias=bias))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], mean, std, bias=bias))

    return np.array(ret_normal), np.array(ret_mutant)
