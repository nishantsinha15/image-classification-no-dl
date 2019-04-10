# open the train file
import csv
import cv2
import os
import pickle
import time
from collections import defaultdict

import mahotas as mahotas
import matplotlib.pyplot as plt
from pandas.tests.groupby.test_value_counts import bins
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from skimage import color
from skimage import io
import numpy as np


def preprocess(image):
    # img = color.rgb2gray(img)
    # import scipy.misc
    # img = scipy.misc.imresize(img, (32, 32))
    # img = img.flatten()
    if image.shape == (64,64):
        temp = np.array([[1, 2], [3, 4]])
        image = np.stack((temp,) * 3, axis=-1)
    if image.shape != (64,64,3):
        print("here ", image.shape)
    global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])
    return global_feature


def fd_hu_moments(image):
    # if image.shape[2] == 3:
    if image.shape == (64,64,3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    # if image.shape[2] == 3:
    if image.shape == (64,64,3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(image).mean(axis=0)
    return haralick


def fd_histogram(image, mask=None):
    bins = 8
    # convert the image to HSV color-space
    # if image.shape == (64, 64, 3):
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     print("image.shape")
    # else:
    #     print("ignored")
    plt.imsave('this.png', image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()


def get_mean(data):
    mean = data.sum(axis=0) / data.shape[0]
    return mean


def get_covariance_matrix(data ):
    mean = get_mean(data)
    mean_substracted_data = data - mean
    # print("Mean Substracted data shape= ", mean_substracted_data.shape)
    cov = np.dot(np.transpose(mean_substracted_data), mean_substracted_data)
    cov /= (data.shape[0] - 1)
    # print("Covaraince matrix shape= ", cov.shape)
    return cov


def do_eigenvalue_decomposition(matrix):
    w, v = np.linalg.eig(matrix)
    return w, v
