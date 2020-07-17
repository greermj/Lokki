import numpy as np

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from lokki.feature_transform import FeatureTransformChoice

class MutualInformation(FeatureTransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape
        self.step_size = int(dataset_shape[1] * 0.30)
        self.grid = [{'k' : x} for x in np.arange(1, dataset_shape[1], step = self.step_size)]

    def fit(self, hyperparameters, X, y):
        self.mutual = SelectKBest(mutual_info_classif, **hyperparameters).fit(X,y)

    def transform(self, X, y):
        return self.mutual.transform(X)

    def get_name(self):
        return 'Mutual_Information'

    def hyperparameter_grid(self):
        return self.grid
