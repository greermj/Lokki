import sys

# Models 
from lokki.model import AdaBoost
from lokki.model import GradientBoosting
from lokki.model import RandomForest

from lokki.model import LogisticRegressionModel
from lokki.model import RidgeClassifierModel
from lokki.model import SVM

from lokki.model import DecisionTree
from lokki.model import ExtraTree

from lokki.model import LDA
from lokki.model import QDA

# Data Transforms 
from lokki.data_transform import NoPreprocessing
from lokki.data_transform import Log
from lokki.data_transform import ZScore

# Feature Tranforms 
from lokki.feature_transform import FactorAnalysis
from lokki.feature_transform import ICA
from lokki.feature_transform import NMF
from lokki.feature_transform import PCA

from lokki.feature_transform import ChiSquare
from lokki.feature_transform import MutualInformation
from lokki.feature_transform import HFE
from lokki.feature_transform import Void


class PipelineComponents:

    def __init__(self, dataset_shape, taxonomy = None):

        self.dataset_shape = dataset_shape
        self.taxonomy      = taxonomy 


    def get_component(self, name, component_type):

        if component_type.strip().lower() == 'data_transform':
            if name.lower() == 'none':
                return NoPreprocessing()
            elif name.lower() == 'log':
                return Log()
            elif name.lower() == 'zscore':
                return ZScore()
            else:
                sys.exit('ERROR: ' + ' Could not find data transform method "' + name + '"')

        if component_type.strip().lower() == 'feature_transform':
            if name.lower() == 'none':
                return Void(self.dataset_shape)
            elif name.lower() == 'chi_square':
                return ChiSquare(self.dataset_shape)
            elif name.lower() == 'mutual_information':
                return MutualInformation(self.dataset_shape)
            elif name.lower() == 'hfe':
                return HFE(self.dataset_shape, self.taxonomy)
            elif name.lower() == 'factor':
                return FactorAnalysis(self.dataset_shape)
            elif name.lower() == 'ica':
                return ICA(self.dataset_shape)
            elif name.lower() == 'nmf':
                return NMF(self.dataset_shape)
            elif name.lower() == 'pca':
                return PCA(self.dataset_shape)
            else:
                sys.exit('ERROR: ' + ' Could not find feature transform method "' + name + '"')

        if component_type.strip().lower() == 'model':
            if name.lower() == 'random_forest':
                return RandomForest()
            elif name.lower() == 'decision_tree':
                return DecisionTree()
            elif name.lower() == 'lda':
                return LDA()
            if name.lower() == 'qda':
                return QDA()
            if name.lower() == 'extra_tree':
                return ExtraTree()
            if name.lower() == 'logistic_regression':
                return LogisticRegressionModel()
            if name.lower() == 'ridge':
                return RidgeClassifierModel()
            if name.lower() == 'adaboost':
                return AdaBoost()
            if name.lower() == 'gradient_boosting':
                return GradientBoosting()
            if name.lower() == 'svm':
                return SVM()
            else:
                sys.exit('ERROR: ' + ' Could not find model "' + name + '"')

