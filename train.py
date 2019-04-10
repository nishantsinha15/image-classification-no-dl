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
        model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
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

    def main(self, trained=False):
        X, Y = self.get_data()
        if not trained:
            model = self.rforest(X, Y)
            # model = adaboost()
            # model = svm_linear(X, Y)
        else:
            model = self.load_model()
        t = time.time()
        test.run_test(model)
        print("Model tested in ", time.time() - t)

    def get_data(self):
        t = time.time()
        X, Y = utils.load_train_data()
        print("data loaded in ", time.time() - t)
        print("Before ",X.shape)
        return X, Y
        # t = time.time()
        # self.pca = Pca.PCA(X, 0.95)
        # print("Pca took ", time.time() - t)
        # t = time.time()
        # X = self.pca.get_projection(X)
        # print("Train Projection took", time.time() - t)
        # print("After ", X.shape)
        # return X, Y
        # test_files, X_test = load_test_data()
        # X_test = pca.get_projection(X_test)
        # print("Test data pca'ed")
        # print(X.shape, X_test.shape)
        # return X, Y, test_files, X_test




my_model = Model()
my_model.main()
# get_data()
