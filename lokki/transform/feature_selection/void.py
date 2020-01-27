from lokki.transform import TransformChoice

class Void(TransformChoice):

    def __init__(self, dataset_shape):
        self.dataset_shape = dataset_shape

    def fit_transform(self, hyperparameters, X, y):
        pass

    def get_name(self):
        return 'Void'

    def hyperparameter_grid(self):
        return None
