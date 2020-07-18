import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

class DataTransformationChoice(object, metaclass = ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y = None):
        pass

    @abstractmethod
    def transform(self, X, y = None):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def hyperparameter_grid(self):
        pass
