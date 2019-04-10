import csv
import pickle
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np
import test
import utils
import Pca


class Model:

    def svm_linear(self, X, Y):
        model = SVC(kernel="linear", C=0.025)
        t = time.time()
        model.fit(X, Y)
        print("Model trained in ", time.time() - t)
        print("Score = ", model.score(X, Y))
        self.save_model(model)
        return model

    def rforest(self, X, Y):
        model = RandomForestClassifier(max_depth=5, n_estimators=500, max_features=1)
        t = time.time()
        model.fit(X, Y)
        print("Model trained in ", time.time() - t)
        print("Score = ", model.score(X, Y))
        self.save_model(model)
        return model

    def adaboost(self, X, Y):
        model = AdaBoostClassifier()
        t = time.time()
        model.fit(X, Y)
        print("Model trained in ", time.time() - t)
        print(model.score(X, Y))
        self.save_model(model, 'adaboost')
        return model

    def save_model(self, model, f_name='dtree.pkl'):
        with open(f_name, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, name='dtree.pkl'):
        with open(name, 'rb') as f:
            model = pickle.load(f)
        return model
