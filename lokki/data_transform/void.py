from lokki.data_transform import DataTransformationChoice

class NoPreprocessing(DataTransformationChoice):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        pass

    def transform(self, X, y = None):
        return X

    def get_name(self):
        return 'No_Data_Transform'

    def hyperparameter_grid(self):
        return None
