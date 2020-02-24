import numpy as np
import sklearn as sk

from lokki.transform import TransformChoice

class ICA(TransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape
        self.step_size = int(dataset_shape[1] * 0.20)
        self.grid = [{'n_components' : x, 'max_iter' : 100, 'tol' : 0.1} for x in np.arange(dataset_shape[1] - 1, step = self.step_size) + 1]

    def fit(self, hyperparameters, X, y):
        self.ica = sk.decomposition.FastICA(**hyperparameters).fit(X)

    def transform(self, X, y):
        return self.ica.transform(X)

    def get_name(self):
        return 'ICA'

    def hyperparameter_grid(self):
        return self.grid
