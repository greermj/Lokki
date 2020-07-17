import numpy as np
import sklearn as sk

from lokki.feature_transform import FeatureTransformChoice

class FactorAnalysis(FeatureTransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape
        self.step_size = int(dataset_shape[1] * 0.30)
        self.grid = [{'n_components' : x} for x in np.arange(1, dataset_shape[1], step = self.step_size)]

    def fit(self, hyperparameters, X, y):
        self.factor = sk.decomposition.FactorAnalysis(**hyperparameters).fit(X)

    def transform(self, X, y):
        return self.factor.transform(X)

    def get_name(self):
        return 'Factor_Analysis'

    def hyperparameter_grid(self):
        return self.grid
