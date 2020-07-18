from lokki.feature_transform import FeatureTransformChoice

class Void(FeatureTransformChoice):

    def __init__(self, dataset_shape, parameters):
        self.dataset_shape = dataset_shape
        self.parameters = parameters

    def fit(self, hyperparameters, X, y):
        pass

    def transform(self, X, y = None):
        return X

    def get_name(self):
        return 'No_Feature_Transform'

    def hyperparameter_grid(self):
        return None
