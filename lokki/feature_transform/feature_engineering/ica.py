import numpy as np
import sklearn as sk

from lokki.feature_transform import FeatureTransformChoice

class ICA(FeatureTransformChoice):

    def __init__(self, dataset_shape, parameters):
        self.dataset_shape = dataset_shape
        self.parameters = parameters
        self.step_size = int(dataset_shape[1] * 0.15)
        self.grid = [{'n_components' : x, 'max_iter' : 100, 'tol' : 0.1} for x in np.arange(1, int(dataset_shape[1] / self.parameters['num_folds']), step = self.step_size)]

    def fit(self, hyperparameters, X, y):
        self.ica = sk.decomposition.FastICA(**hyperparameters).fit(X)

    def transform(self, X, y = None):
        return self.ica.transform(X)

    def get_name(self):
        return 'ICA'

    def hyperparameter_grid(self):
        return self.grid
