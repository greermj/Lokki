import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

class FeatureTransformChoice(object, metaclass = ABCMeta):

    def __init__(self, dataset_shape):
        pass

    @abstractmethod
    def fit(self, hyperparameters, X, y):
        pass

    @abstractmethod
    def transform(self, X, y):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def hyperparameter_grid(self):
        pass
