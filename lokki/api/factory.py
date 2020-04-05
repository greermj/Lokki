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
from lokki.process import NoPreprocessing 
from lokki.process import Log 
from lokki.process import ZScore

# Tranforms 
from lokki.transform import FactorAnalysis
from lokki.transform import ICA
from lokki.transform import NMF
from lokki.transform import PCA

from lokki.transform import ChiSquare
from lokki.transform import MutualInformation
from lokki.transform import Void

# Visualizations 
from lokki.visualize import Stacked 
from lokki.visualize import Enrichment 

def configure(**kwargs):
    return AnalysisFactory(kwargs['dataset'], kwargs['target_name'], kwargs['pre_processing'], kwargs['transforms'], kwargs['models'], kwargs['metric'])

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
        plot = Stacked(analysis_object)
    if kwargs['plot_type'].lower() == 'enrichment':
        plot = Enrichment(analysis_object)
        
    return plot.run(kwargs['output_filename'])

class AnalysisFactory:
    """Builds analysis objects"""

    def __init__(self, dataset, target_name, pre_processing, transforms, models, metric):
        """AnalysisFactory init

        :param dataset: pandas dataframe containing data to analyze 
        :param target_name: target name that is present within the dataset
        :param transforms: list of transforms to search 
        :param models: list of models to search
        :param metric: training metric (eg auc, precision, etc)
        """

        self.dataset = dataset
        self.dataset_shape = dataset.shape
        self.model_transform_tuples = list(product(transforms, models))
        self.parameters  = {'target_name' : target_name, 'metric' : metric, 'num_iterations' : 5, 'num_folds' : 5}

        self.analysis_runs = []

        for process, transform, model in list(product(pre_processing, transforms, models)):
            #print(process + '\t' + transform + '\t' + model)

            analysis_process = None
            analysis_transform = None
            analysis_model = None
    
            # Preprocessing Strategy 
            if process.lower() == 'none':
                analysis_process = NoPreprocessing()
            elif process.lower() == 'log':
                analysis_process = Log()
            elif process.lower() == 'zscore':
                analysis_process = ZScore()
            else:
                print('Error: Preprocessing method not found')

            # Feature Engineering Strategies 
            if transform.lower() == 'factor':
                analysis_transform = FactorAnalysis(self.dataset_shape)
            elif transform.lower() == 'ica':
                analysis_transform = ICA(self.dataset_shape)
            elif transform.lower() == 'nmf':
                analysis_transform = NMF(self.dataset_shape)
            elif transform.lower() == 'pca':
                analysis_transform = PCA(self.dataset_shape)

            # Feature Selection Strategies
            elif transform.lower() == 'none':
                analysis_transform = Void(self.dataset_shape)
            elif transform.lower() == 'chi_square':
                analysis_transform = ChiSquare(self.dataset_shape)
            elif transform.lower() == 'mutual_information':
                analysis_transform = MutualInformation(self.dataset_shape)
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

            self.analysis_runs.append(ModelTransformAnalysis(analysis_process, analysis_transform, analysis_model, self.parameters))

    def run(self):

        self.results = []
        
        for i, analysis in enumerate(self.analysis_runs):
            current_process   = '_'.join(analysis.process_instance.get_name().lower().split(' '))
            current_transform = '_'.join(analysis.transform_instance.get_name().lower().split(' '))
            current_model     = '_'.join(analysis.model_instance.get_name().lower().split(' '))
            print('Analyzing: ' + current_process + '_' + current_transform + '_' + current_model)
            self.results.append({'key'   : (current_process.strip().lower(), current_transform.strip().lower(), current_model.strip().lower()), 
                                 'value' : analysis.get_performance(self.dataset)})

        return AnalysisObject(self.results, self.parameters['metric'])
