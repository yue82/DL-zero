# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from grads import numerical_diff


def plt_tangent(x, y, tangent, save=False, filename='test.png'):
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.plot(x, y)
    plt.plot(x, tangent)

    if save:
        plt.savefig(filename)
    else:
        plt.show()


def tangent_line(f, x, a):
    b = f(x) - a * x  # f(t) = a * x + b
    return lambda x: a * x + b


def func1(x):
    return 0.01*x**2 + 0.1*x


if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y1 = func1(x)

    t = 5.0
    d = numerical_diff(func1, t)
    print('nunmerical diff(x={}): {}'.format(t, d))

    tl = tangent_line(func1, t, d)
    y2 = tl(x)

    plt_tangent(x, y1, y2, save=True, filename='func1_tangent.png')
