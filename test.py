import csv
import os
import utils
import matplotlib.pyplot as plt



def write_to_csv(file_names, y):
    with open('test_output_rforest.csv', mode='w') as f:
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
        if x.shape != (64,64,3):
            print(x.shape)
        # x = utils.preprocess(x)
        X.append(x)
    return test_files, X


