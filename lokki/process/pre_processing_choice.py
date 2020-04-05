import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

class PreProcessingChoice(object, metaclass = ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
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
