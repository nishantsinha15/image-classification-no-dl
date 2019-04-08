import csv
import os
import utils
import matplotlib.pyplot as plt


def write_to_csv(file_names, y):
    with open('test_output.csv', mode='w') as f:
        test_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_writer.writerow(['Id', 'Category'])

        for i, name in enumerate(file_names):
            test_writer.writerow([name, str(y[i])])


def load_data():
    test_files = os.listdir('data/sml_test')
    X = []
    for file in test_files:
        name = 'data/sml_test/' + file
        x = plt.imread(name)
        x = utils.preprocess(x)
        X.append(x)
    return test_files, X


def run_test(model):
    test_files, X = load_data()
    pred = model.predict(X)
    write_to_csv(test_files, pred)