# -*- coding: utf-8 -*-

import numpy as np


def step_func(x):
    y = x > 0
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_func(x):
    return x


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):
    c = np.max(x)
    expx = np.exp(x - c)
    sum_expx = np.sum(expx)
    return expx / sum_expx
