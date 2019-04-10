# open the train file
import cv2

import mahotas as mahotas
import matplotlib.pyplot as plt
import numpy as np


def preprocess(image):
    # img = color.rgb2gray(img)
    # import scipy.misc
    # img = scipy.misc.imresize(img, (32, 32))
    # img = img.flatten()
    if image.shape == (64, 64):
        image = np.dstack((image, image, image))

    if image.shape != (64,64,3):
        print("Caution ", image.shape)
    global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])
    # print(global_feature.shape, flush=True)
    return global_feature


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(image).mean(axis=0)
    return haralick


def fd_histogram(image, mask=None):
    bins = 8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()
