# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from activation_funcs import step_func, sigmoid, ReLU, identity_func, softmax


def activation_funcs_graph(funcs):
    plt.title('activateion_funcs')
    x = np.arange(-5.0, 5.0, 0.1)
    for func in funcs:
        y = func(x)
        plt.plot(x, y, label=func.__name__)
        plt.ylim(-0.1, 1.1)
    plt.legend()
    # plt.show()
    plt.savefig('activateion_funcs.png')


if __name__ == '__main__':
    funcs = [step_func, sigmoid, ReLU, identity_func, softmax]
    activation_funcs_graph(funcs)
