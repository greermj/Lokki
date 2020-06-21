from itertools import product 

from lokki.analyze import ModelTransformAnalysis 
from lokki.analyze import AnalysisObject

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

# Preprocessing
from lokki.data_transform import NoPreprocessing 
from lokki.data_transform import Log 
from lokki.data_transform import ZScore

# Tranforms 
from lokki.feature_transform import FactorAnalysis
from lokki.feature_transform import ICA
from lokki.feature_transform import NMF
from lokki.feature_transform import PCA

from lokki.feature_transform import ChiSquare
from lokki.feature_transform import MutualInformation
from lokki.feature_transform import HFE
from lokki.feature_transform import Void

# Visualizations 
from lokki.visualize import Stacked 
from lokki.visualize import Enrichment 
from lokki.visualize import Hybrid 

def configure(**kwargs):
    return AnalysisFactory(kwargs['dataset'], kwargs['target_name'], kwargs['data_transforms'], kwargs['feature_transforms'], kwargs['models'], kwargs['metric'], kwargs['taxonomy'] if 'taxonomy' in kwargs else None)

def custom(**kwargs):

    results        = []
    data           = kwargs['dataset']
    scoring_metric_name = kwargs['scoring_metric_name']

    for i in range(0, len(data)):
        current_keys = tuple([x for x in data.columns.values[(data.iloc[i] == 1).values] if x.strip().lower() != 'sample' and x.strip().lower() != 'score'])
        results.append({'key'   : current_keys,
                        'value' : data.iloc[i]['score']})

    return AnalysisObject(results, scoring_metric_name)

def plot(**kwargs):

    analysis_object = kwargs['analysis_object']
        
    plot = None
    
    if kwargs['plot_type'].lower() == 'stacked':
        plot = Stacked(analysis_object, kwargs)
    if kwargs['plot_type'].lower() == 'enrichment':
        plot = Enrichment(analysis_object, kwargs)
    if kwargs['plot_type'].lower() == 'hybrid':
        plot = Hybrid(analysis_object, kwargs)

    return plot.run()

class AnalysisFactory:
    """Builds analysis objects"""

    def __init__(self, dataset, target_name, data_transforms, feature_transforms, models, metric, taxonomy):
        """AnalysisFactory init

        :param dataset: pandas dataframe containing data to analyze 
        :param target_name: target name that is present within the dataset
        :param transforms: list of transforms to search 
        :param models: list of models to search
        :param metric: training metric (eg auc, precision, etc)
        """

        self.dataset = dataset
        self.dataset_shape = dataset.shape
        self.taxonomy = taxonomy
        self.model_transform_tuples = list(product(feature_transforms, models))
        self.parameters  = {'target_name' : target_name, 'metric' : metric, 'num_iterations' : 5, 'num_folds' : 5}

        self.analysis_runs = []

        for data_transform, feature_transform, model in list(product(data_transforms, feature_transforms, models)):

            analysis_data_transform = None
            analysis_feature_transform = None
            analysis_model = None
    
            # Data Transformation Strategy
            if data_transform.lower() == 'none':
                analysis_data_transform = NoPreprocessing()
            elif data_transform.lower() == 'log':
                analysis_data_transform = Log()
            elif data_transform.lower() == 'zscore':
                analysis_data_transform = ZScore()
            else:
                print('Error: Preprocessing method not found')

            # Feature Engineering Strategies 
            if feature_transform.lower() == 'factor':
                analysis_feature_transform = FactorAnalysis(self.dataset_shape)
            elif feature_transform.lower() == 'ica':
                analysis_feature_transform = ICA(self.dataset_shape)
            elif feature_transform.lower() == 'nmf':
                analysis_feature_transform = NMF(self.dataset_shape)
            elif feature_transform.lower() == 'pca':
                analysis_feature_transform = PCA(self.dataset_shape)

            # Feature Selection Strategies
            elif feature_transform.lower() == 'none':
                analysis_feature_transform = Void(self.dataset_shape)
            elif feature_transform.lower() == 'chi_square':
                analysis_feature_transform = ChiSquare(self.dataset_shape)
            elif feature_transform.lower() == 'mutual_information':
                analysis_feature_transform = MutualInformation(self.dataset_shape)
            elif feature_transform.lower() == 'hfe':
                analysis_feature_transform = HFE(self.dataset_shape, self.taxonomy)
            else:
                print('Error: Transform method not found')

            # Modeling Strategies 
            if model.lower() == 'random_forest':
                analysis_model = RandomForest()
            elif model.lower() == 'decision_tree':
                analysis_model = DecisionTree()
            elif model.lower() == 'lda':
                analysis_model = LDA()
            elif model.lower() == 'qda':
                analysis_model = QDA()
            elif model.lower() == 'extra_tree':
                analysis_model = ExtraTree()
            elif model.lower() == 'logistic_regression':
                analysis_model = LogisticRegressionModel()
            elif model.lower() == 'ridge':
                analysis_model = RidgeClassifierModel()
            elif model.lower() == 'adaboost':
                analysis_model = AdaBoost()
            elif model.lower() == 'gradient_boosting':
                analysis_model = GradientBoosting()
            elif model.lower() == 'svm':
                analysis_model = SVM()
            else:
                print('Error: Model method not found')

            self.analysis_runs.append(ModelTransformAnalysis(analysis_data_transform, analysis_feature_transform, analysis_model, self.parameters))

    def run(self):

        self.results = []
        
        for i, analysis in enumerate(self.analysis_runs):
            current_data_transform   = '_'.join(analysis.data_transform_instance.get_name().lower().split(' '))
            current_feature_transform = '_'.join(analysis.feature_transform_instance.get_name().lower().split(' '))
            current_model     = '_'.join(analysis.model_instance.get_name().lower().split(' '))
            print('Analyzing: ' + current_data_transform + '_' + current_feature_transform + '_' + current_model)
            self.results.append({'key'   : (current_data_transform.strip().lower(), current_feature_transform.strip().lower(), current_model.strip().lower()), 
                                 'value' : analysis.get_performance(self.dataset)})

        return AnalysisObject(self.results, self.parameters['metric'])
