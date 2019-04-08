# open the train file
import csv
import os
import pickle
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from skimage import color
from skimage import io
import numpy as np
import utils


def read_csv(file_name='data/sml_train.csv'):
    x_files = []
    y = []
    count = defaultdict(int)
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            x_files.append(row[0])
            y.append(int(row[1]))
            count[int(row[1])] += 1
    return x_files, y


def read_images(x_files, prefix='data/sml_train/'):
    X = []
    for file in x_files:
        name = prefix + file
        x = plt.imread(name)
        x = utils.preprocess(x)
        X.append(x)
    return np.asarray(X)


def load_data():
    x_files, Y = read_csv()
    X = read_images(x_files)
    return X, np.asarray(Y)


t = time.time()
X, Y = load_data()
print("data loaded in ", time.time() - t)
print(X.shape, Y.shape)