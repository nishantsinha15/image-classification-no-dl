# open the train file
import cv2

import mahotas as mahotas
import matplotlib.pyplot as plt
import numpy as np


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
