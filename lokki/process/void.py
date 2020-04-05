from lokki.process import PreProcessingChoice

class NoPreprocessing(PreProcessingChoice):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X, y):
        return X

    def get_name(self):
        return 'No_Preprocessing'

    def hyperparameter_grid(self):
        return None
