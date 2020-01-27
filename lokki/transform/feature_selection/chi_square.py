from sklearn.feature_selection import chi2

from lokki.transform import TransformChoice

class ChiSquare(TransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape

    def fit_transform(self, hyperparameters, X, y):
        pass

    def get_model_name(self):
        return 'Chi Square'

    def hyperparameter_grid(self):
        pass
