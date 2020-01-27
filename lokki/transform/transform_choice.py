import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

class TransformChoice(object, metaclass = ABCMeta):

    def __init__(self, dataset_shape):
        pass

    @abstractmethod
    def fit_transform(self, hyperparameters, X, y):
        pass

    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def hyperparameter_grid(self):
        pass
