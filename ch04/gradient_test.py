# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from grads import numerical_gradient


def plt_grad(x0, x1, grad, save=False, filename='test.png'):
    plt.figure()
    plt.quiver(x0, x1, -grad[0], -grad[1],
               angles="xy",color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()

    if save:
        plt.savefig(filename)
    else:
        plt.show()


def func2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


if __name__ == '__main__':
    x0_range = np.arange(-2.0, 2.25, 0.25)
    x1_range = np.arange(-2.0, 2.25, 0.25)

    x0, x1 = np.meshgrid(x0_range, x1_range)
    x0 = x0.flatten()
    x1 = x1.flatten()

    grad = numerical_gradient(func2, np.array([x0, x1]))
    plt_grad(x0, x1, grad, save=True, filename='func2_grad.png')
