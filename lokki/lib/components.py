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

    def __init__(self, dataset_shape, parameters, taxonomy = None):

        self.dataset_shape = dataset_shape
        self.parameters    = parameters
        self.taxonomy      = taxonomy 


    def get_component(self, name, component_type):

        if component_type.strip().lower() == 'data_transform':
            if name.lower() == 'none' or name.lower() == 'no_data_transform':
                return NoPreprocessing()
            elif name.lower() == 'log':
                return Log()
            elif name.lower() == 'zscore':
                return ZScore()
            else:
                sys.exit('ERROR: ' + ' Could not find data transform method "' + name + '"')

        if component_type.strip().lower() == 'feature_transform':
            if name.lower() == 'none' or name.lower() == 'no_feature_transform':
                return Void(self.dataset_shape, self.parameters)
            elif name.lower() == 'chi_square':
                return ChiSquare(self.dataset_shape, self.parameters)
            elif name.lower() == 'mutual_information':
                return MutualInformation(self.dataset_shape, self.parameters)
            elif name.lower() == 'hfe':
                return HFE(self.dataset_shape, self.taxonomy)
            elif name.lower() == 'factor_analysis':
                return FactorAnalysis(self.dataset_shape, self.parameters)
            elif name.lower() == 'ica':
                return ICA(self.dataset_shape, self.parameters)
            elif name.lower() == 'nmf':
                return NMF(self.dataset_shape, self.parameters)
            elif name.lower() == 'pca':
                return PCA(self.dataset_shape, self.parameters)
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
            if name.lower() == 'ridge_regression':
                return RidgeClassifierModel()
            if name.lower() == 'adaboost':
                return AdaBoost()
            if name.lower() == 'gradient_boosting':
                return GradientBoosting()
            if name.lower() == 'svm':
                return SVM()
            else:
                sys.exit('ERROR: ' + ' Could not find model "' + name + '"')

    # Description: Returns a dictionary mapping the name to the component type
    def get_name_to_component_map(self):

        def no_space(string):
            return '_'.join(string.split(' '))

        return { no_space(NoPreprocessing.get_name('').lower())         : 'data_transform',
                 no_space(Log.get_name('').lower())                     : 'data_transform',
                 no_space(ZScore.get_name('').lower())                  : 'data_transform',

                 no_space(Void.get_name('').lower())                    : 'feature_transform',
                 no_space(ChiSquare.get_name('').lower())               : 'feature_transform',
                 no_space(MutualInformation.get_name('').lower())       : 'feature_transform',
                 no_space(HFE.get_name('').lower())                     : 'feature_transform',
                 no_space(FactorAnalysis.get_name('').lower())          : 'feature_transform',
                 no_space(ICA.get_name('').lower())                     : 'feature_transform',
                 no_space(NMF.get_name('').lower())                     : 'feature_transform',
                 no_space(PCA.get_name('').lower())                     : 'feature_transform',

                 no_space(RandomForest.get_name('').lower())            : 'model',
                 no_space(DecisionTree.get_name('').lower())            : 'model',
                 no_space(LDA.get_name('').lower())                     : 'model',
                 no_space(QDA.get_name('').lower())                     : 'model',
                 no_space(ExtraTree.get_name('').lower())               : 'model',
                 no_space(LogisticRegressionModel.get_name('').lower()) : 'model',
                 no_space(RidgeClassifierModel.get_name('').lower())    : 'model',
                 no_space(AdaBoost.get_name('').lower())                : 'model',
                 no_space(GradientBoosting.get_name('').lower())        : 'model',
                 no_space(SVM.get_name('').lower())                     : 'model'}
