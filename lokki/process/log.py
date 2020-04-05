import numpy as np
import pandas as pd
import sklearn as sk

from lokki.process import PreProcessingChoice

class Log(PreProcessingChoice):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X, y):
        return pd.DataFrame(np.log(X.values + 1))

    def get_name(self):
        return 'Log_Preprocessing'

    def hyperparameter_grid(self):
        return self.grid
