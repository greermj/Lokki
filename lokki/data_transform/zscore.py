import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from lokki.data_transform import DataTransformationChoice

class ZScore(DataTransformationChoice):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.zscore = StandardScaler().fit(X)

    def transform(self, X, y):
        data = pd.DataFrame(self.zscore.transform(X), columns = X.columns.values)
        data[data > 3] = 3
        data[data <= -3] = -3
        return data.values + 3

    def get_name(self):
        return 'ZScore_Preprocessing'

    def hyperparameter_grid(self):
        return self.grid