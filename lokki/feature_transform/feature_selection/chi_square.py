import numpy as np

from sklearn.feature_selection import SelectKBest, chi2

from lokki.feature_transform import FeatureTransformChoice

class ChiSquare(FeatureTransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape
        self.step_size = int(dataset_shape[1] * 0.20)
        self.grid = [{'k' : x} for x in np.arange(1, dataset_shape[1] - 1, step = self.step_size)]

    def fit(self, hyperparameters, X, y):
        self.chi = SelectKBest(chi2, **hyperparameters).fit(X,y)

    def transform(self, X, y):
        return self.chi.transform(X)

    def get_name(self):
        return 'Chi_Square'

    def hyperparameter_grid(self):
        return self.grid
