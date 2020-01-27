import numpy as np
import sklearn as sk

from lokki.transform import TransformChoice

class PCA(TransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape
        self.step_size = int(dataset_shape[1] * 0.20)
        self.grid = [{'n_components' : x} for x in np.arange(dataset_shape[1] - 1, step = self.step_size) + 1]

    def fit_transform(self, hyperparameters, X, y):
        return sk.decomposition.PCA(**hyperparameters).fit_transform(X)

    def get_name(self):
        return 'PCA'

    def hyperparameter_grid(self):
        return self.grid
