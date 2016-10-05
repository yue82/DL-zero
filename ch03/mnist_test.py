# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image


def img_show(img, save=False, filename='test.png'):
    pil_img = Image.fromarray(np.uint8(img))
    if save:
        pil_img.save(filename)
    else:
        pil_img.show()


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)

    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)

    img = x_train[0]
    label = t_train[0]
    print(label)
    print(img.shape)

    img = img.reshape(28, 28)
    print(img.shape)
    img_show(img, save=True, filename='mnist_input_test.png')
