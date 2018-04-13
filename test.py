# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sys import argv
import sys
import random


def gaussianNoisy(im, mean, sigma):
    """
    对图像做高斯噪音处理
    :param im: 单通道图像
    :param mean: 偏移量
    :param sigma: 标准差
    :return:
    """
    for _i in range(len(im)):
        im[_i] += random.gauss(mean, sigma)
    return im

datapath = "C:/Users/Wang Jingyuan/Desktop/把.set"
file = open(datapath, 'rb')
x = np.fromfile(file, dtype = np.ubyte).reshape(1000,48*48)
for i in range(1000):
    img_gau = gaussianNoisy(x[i], mean=0.2, sigma=0.3)
    image6 = img_gau.reshape([48, 48])
    plt.imshow(image6, cmap='gray')
    plt.show()


