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
    print(len(x_files), len(set(y)))
    return x_files, y


def read_images(x_files, prefix = 'data/sml_train/'):
    X = []
    X_test = []
    for file in x_files:
        name = prefix + file
        x = plt.imread(name)
        x = preprocess(x)
        X.append(x)

    test_files = os.listdir('data/sml_test')
    for file in test_files:
        name = 'data/sml_test/' + file
        x = plt.imread(name)
        x = preprocess(x)
        X_test.append(x)
    print(len(X), len(X_test))
    return X, X_test, test_files


def model(X, Y):
    print(X.shape, Y.shape)
    # Create classifiers
    # lr = LogisticRegression(solver='lbfgs')
    # lr.fit(X, Y)
    # print("Logistic Regression trained in ", lr.score(X, Y))
    #
    # gnb = GaussianNB()
    # gnb.fit(X,Y)
    # print("Gaussian NB trained in ", gnb.score(X, Y))

    t = time.time()
    svc = LinearSVC(C=1.0)
    svc.fit(X,Y)
    filename = 'svc_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print("Linear SVC trained ", svc.score(X,Y), time.time() - t)

    t = time.time()
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X, Y)
    filename = 'rf_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print('Random forest trained in ', rfc.score(X,Y), time.time() - t )


def predict(X_test, test_files):
    model = pickle.load(open('svc_model.sav', 'rb'))
    Y = model.predict(X_test)



x, Y = read_csv()
X, X_test, test_files = read_images(x)
# model(np.asarray(X), np.asarray(Y))
predict(X, Y)