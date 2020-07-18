import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from lokki.data_transform import DataTransformationChoice

class ZScore(DataTransformationChoice):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        self.zscore = StandardScaler().fit(X.astype(float))

    def transform(self, X, y = None):
        data = pd.DataFrame(self.zscore.transform(X.astype(float)), columns = X.columns.values)
        data[data > 3] = 3
        data[data <= -3] = -3
        return (data + 3).copy()

    def get_name(self):
        return 'ZScore'

    def hyperparameter_grid(self):
        return self.grid
