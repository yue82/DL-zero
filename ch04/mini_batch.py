# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
    return x_train, t_train, x_test, t_test


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = get_data()
    print(x_train.shape)
    print(t_train.shape)
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    print(batch_mask)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(x_batch.shape)
    print(t_batch.shape)
