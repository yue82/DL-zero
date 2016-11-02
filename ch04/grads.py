# -*- coding: utf-8 -*-

import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f, xs):
    if xs.ndim == 1:
        return _numerical_gradient_no_batch(f, xs)
    else:
        grad = np.zeros_like(xs)

        for i, x in enumerate(xs):
            grad[i] = _numerical_gradient_no_batch(f, x)

        return grad


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_xi = x[i]
        x[i] = tmp_xi + h
        fxph = f(x)

        x[i] = tmp_xi - h
        fxmh = f(x)

        grad[i] = (fxph - fxmh) / (2*h)

        x[i] = tmp_xi
    return grad
