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


def preprocess(img):
    img = color.rgb2gray(img)
    img = img.flatten()
    return img
