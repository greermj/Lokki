import numpy as np
import pandas as pd
import sklearn as sk

from lokki.data_transform import DataTransformationChoice

class Log(DataTransformationChoice):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        pass

    def transform(self, X, y = None):
        return pd.DataFrame(np.log(X.values + 1), columns = X.columns.values)

    def get_name(self):
        return 'Log'

    def hyperparameter_grid(self):
        return self.grid
