import csv
import os
import pickle
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from skimage import color
from skimage import io
import numpy as np
import utils
import test
import Pca


def rforest(X, Y):
    model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    t = time.time()
    model.fit(X, Y)
    print("Model trained in ", time.time() - t)
    print("Score = ", model.score(X, Y))
    save_model(model)
    return model


def adaboost(X, Y):
    model = AdaBoostClassifier()
    t = time.time()
    model.fit(X, Y)
    print("Model trained in ", time.time() - t)
    print(model.score(X, Y))
    save_model(model, 'adaboost')
    return model


def save_model(model):
    with open('my_dumped_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)


def load_model(name = 'my_dumped_classifier.pkl'):
    with open(name, 'rb') as f:
        model = pickle.load(f)
    return model


def main(trained = True):
    X, Y, test_files, X_test = get_data()
    if not trained:
        model = rforest(X, Y)
        # model = adaboost()
    else:
        model = load_model()
    t = time.time()
    run_test(model, test_files, X_test)
    print("Model tested in ", time.time()-t)


def get_data():
    t = time.time()
    X, Y = utils.load_train_data()
    print("data loaded in ", time.time() - t)
    t = time.time()
    pca = Pca.PCA(X, 0.95)
    print("Pca took ", time.time() - t)
    t = time.time()
    X = pca.get_projection(X)
    print("Projection took", time.time() - t)
    test_files, X_test = load_test_data()
    X_test = pca.get_projection(X_test)
    print("Test data pca'ed")
    print(X.shape, X_test.shape)
    return X, Y, test_files, X_test



def write_to_csv(file_names, y):
    with open('test_output.csv', mode='w') as f:
        test_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_writer.writerow(['Id', 'Category'])

        for i, name in enumerate(file_names):
            test_writer.writerow([name, str(y[i])])


def load_test_data():
    test_files = os.listdir('data/sml_test')
    X = []
    for file in test_files:
        name = 'data/sml_test/' + file
        x = plt.imread(name)
        x = utils.preprocess(x)
        X.append(x)
    return test_files, X


def run_test(model, test_files, X):
    # test_files, X = load_test_data()
    pred = model.predict(X)
    write_to_csv(test_files, pred)

main()
# get_data()