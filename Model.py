import csv
import pickle
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
import test
import utils
import Pca


class Model:

    def sgd(self, X, Y):
        model = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        t = time.time()
        model.fit(X, Y)
        print("Model trained in ", time.time() - t)
        print("Score = ", model.score(X, Y))
        self.save_model(model, 'mlp')
        return model

    def mlp(self, X, Y):
        model = MLPClassifier((100, 100, 100, 100), verbose=True)
        t = time.time()
        model.fit(X, Y)
        print("Model trained in ", time.time() - t)
        print("Score = ", model.score(X, Y))
        self.save_model(model, 'mlp')
        return model

    def svm_linear(self, X, Y):
        model = SVC(kernel="linear", C=0.025, verbose=True)
        t = time.time()
        model.fit(X, Y)
        print("Model trained in ", time.time() - t)
        print("Score = ", model.score(X, Y))
        self.save_model(model, 'svm_other')
        return model

    def rforest(self, X, Y):
        model = RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=1)
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
