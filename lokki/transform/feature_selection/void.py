from lokki.transform import TransformChoice

class Void(TransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape

    def fit(self, hyperparameters, X, y):
        pass

    def transform(self, X, y):
        return X

    def get_name(self):
        return 'None'

    def hyperparameter_grid(self):
        return None
