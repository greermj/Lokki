import numpy as np
import sklearn as sk

from lokki.feature_transform import FeatureTransformChoice

class PCA(FeatureTransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape
        self.step_size = int(dataset_shape[1] * 0.20)
        self.grid = [{'n_components' : x} for x in np.arange(dataset_shape[1] - 1, step = self.step_size) + 1]

    def fit(self, hyperparameters, X, y):

        # PCA only works if the n_components is less than the min of the # of samples and # of columns 
        if hyperparameters['n_components'] < np.min(X.shape):

            self.pca = sk.decomposition.PCA(**hyperparameters).fit(X)

        else:

        # Select the fewest number of features in all other cases 
            self.pca = sk.decomposition.PCA(**self.grid[0]).fit(X)

    def transform(self, X, y):
        return self.pca.transform(X)

    def get_name(self):
        return 'PCA'

    def hyperparameter_grid(self):
        return self.grid
