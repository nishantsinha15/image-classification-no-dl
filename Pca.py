import numpy as np
import utils
import time
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

class PCA:
    # data.shape = (N, D)
    def __init__(self, data, threshhold_energy):
        self.mean = utils.get_mean(data) # (1, D)
        t = time.time()
        self.covariance_matrix = utils.get_covariance_matrix(data) # (D, D)
        print(self.covariance_matrix.shape, " Cov matrix calculated in ", time.time() - t)
        t = time.time()
        self.eigen_values, self.eigen_vectors = \
            utils.do_eigenvalue_decomposition(self.covariance_matrix)
        print(self.eigen_vectors.shape, self.eigen_values.shape, "Eigen value calculated in ", time.time() - t)
        t = time.time()
        sorted_eigenvalues = []
        for i, val in enumerate(self.eigen_values):
            sorted_eigenvalues.append((val, i))
        sorted_eigenvalues.sort(reverse=True) # (D, D)
        sorted_eigenvectors = self.eigen_vectors.copy()
        for i in range(sorted_eigenvectors.shape[1]):
            sorted_eigenvectors[:,i] = self.eigen_vectors[:,sorted_eigenvalues[i][1]]
        # print("after sorting")
        # print(sorted_eigenvalues)
        # print(sorted_eigenvectors)
        eigen_energy = []
        cum_sum = 0
        for i, val in enumerate(sorted_eigenvalues):
            cum_sum += val[0]
            eigen_energy.append(cum_sum)
        eigen_energy /= eigen_energy[len(sorted_eigenvalues)-1]

        # print("Eigen Energy", eigen_energy)
        thresh = -1

        for i, val in enumerate(eigen_energy):
            if val >= threshhold_energy:
                thresh = i
                break
        if thresh == -1:
            thresh = len(eigen_energy)-1

        outfile = TemporaryFile()
        self.projection_matrix = sorted_eigenvectors[:, :thresh+1] # (D, K)
        np.save(outfile, np.asarray(self.projection_matrix))

    def get_projection(self, data):
        new_data = []
        for x in data:
            projection = np.dot(np.transpose(self.projection_matrix), x) # (D, K) * (N, D)
            new_data.append(projection)
        new_data = np.asarray(new_data)
        # print("New data shape ", new_data.shape)
        return new_data

    def visualize(self, x, label = 'face 1'):
        print("Visualizing ", label)
        x = np.reshape(x, (16,16))
        plt.imshow(x)
        plt.savefig(label + '.png')

    def view_eigen(self):
        for i in range(self.projection_matrix.shape[1]):
            self.visualize(self.projection_matrix[:, i], "face " + str(i))

# y = [[2,1], [3,4], [5,0], [7,6], [9,2]]
# pca = PCA(np.asarray(y), 0.5)
