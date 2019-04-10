import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import Preprocess


# no pre-processing, just the data
class DataLoader:

    def load_test_data(self, reload = False):
        if reload:
            test_files = np.asarray(os.listdir('data/sml_test'))
            X = self.read_images(test_files, 'data/sml_test/')
            np.save('data/test_X.npy', X)
            np.save('data/test_files.npy', test_files)
            return test_files, X
        else:
            test_files = np.load('data/test_files.npy')
            X = np.load('data/test_X.npy')
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
        progress = 0
        for file in x_files:
            progress += 1
            name = prefix + file
            x = plt.imread(name)
            x = Preprocess.preprocess(x)
            X.append(x)
            if progress % 100 == 0:
                print("Processed ", progress)
        return np.asarray(X)

    def load_train_data(self, reload = False):
        if reload:
            x_files, y = self.read_csv()
            X = self.read_images(x_files)
            np.save('data/train_X.npy', X)
            np.save('data/train_y.npy', y)
            return X, np.asarray(y)
        else:
            X = np.load('data/train_X.npy')
            y = np.load('data/train_y.npy')
            return X, y

    def write_to_csv(self, file_names, y):
        with open('test_output_rforest.csv', mode='w') as f:
            test_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            test_writer.writerow(['Id', 'Category'])

            for i, name in enumerate(file_names):
                test_writer.writerow([name, str(y[i])])


# dataLoader = DataLoader()
# X, y = dataLoader.load_train_data()
# print(X.shape, y.shape)

''' 
Problem number 1 : Train has 9836 (64,64,3) data and 164 (64,64) data 
Solved by np.dstack
'''
