import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


# no pre-processing, just the data
class DataLoader:

    '''
    def get_5_fold(self):
        X, y = self.load_train_data()
        print("data loaded")
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        sss.get_n_splits(X, y)
        split = []
        for train_index, test_index in sss.split(X, y):
            print(len(train_index), len(test_index))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            split.append((X_train, y_train, X_test, y_test))
            # do something
        return split
    '''

    def load_test_data(self):
        test_files = os.listdir('data/sml_test')
        X = self.read_images(test_files, 'data/sml_test/')
        return test_files, X

    def read_csv(self, file_name='data/sml_train.csv'):
        x_files = []
        y = []
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                x_files.append(row[0])
                y.append(int(row[1]))
        return x_files, y

    def read_images(self, x_files, prefix='data/sml_train/'):
        X = []
        for file in x_files:
            name = prefix + file
            x = plt.imread(name)
            if x.shape == (64, 64):
                x = np.dstack((x, x, x))
            X.append(x)
        return np.asarray(X)

    def load_train_data(self):
        x_files, Y = self.read_csv()
        X = self.read_images(x_files)
        return X, np.asarray(Y)

    def write_to_csv(self, file_names, y):
        with open('test_output_rforest.csv', mode='w') as f:
            test_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            test_writer.writerow(['Id', 'Category'])

            for i, name in enumerate(file_names):
                test_writer.writerow([name, str(y[i])])


dataLoader = DataLoader()
dataLoader.get_test_data()

''' 
Problem number 1 : Train has 9836 (64,64,3) data and 164 (64,64) data 
Solved by np.dstack
'''
