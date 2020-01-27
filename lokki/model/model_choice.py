import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

class ModelChoice(object, metaclass = ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_name(self):
        pass
