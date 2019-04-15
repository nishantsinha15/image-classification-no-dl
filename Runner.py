import sys

from sklearn.model_selection import StratifiedShuffleSplit
import Model
import DataLoader


def do_testing(model):
    files, X = dataLoader.load_test_data()
    y = model.predict(X)
    dataLoader.write_to_csv(files, y)


def finalize():
    print("Loading Data")
    X, y = dataLoader.load_train_data()
    modelClass = Model.Model()
    rforest = modelClass.rforest(X, y)
    do_testing(rforest)
    sys.exit()


if __name__ == '__main__':
    dataLoader = DataLoader.DataLoader()
    # finalize()
    print("Loading Data")
    X, y = dataLoader.load_train_data()
    modelClass = Model.Model()
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    sss.get_n_splits(X, y)
    i = 0
    modelClass = Model.Model()
    for train_index, test_index in sss.split(X, y):
        print("Fold = ", i)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        i += 1

        svm = modelClass.rforest(X_train, y_train)
        print("Validation Score = ", svm.score(X_test, y_test))
