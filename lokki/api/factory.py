from itertools import product 

from lokki.analyze import ModelTransformAnalysis 

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
    return AnalysisFactory(kwargs['dataset'], kwargs['target_name'], kwargs['transforms'], kwargs['models'], kwargs['metric'])

def custom(**kwargs):

    class AnalysisObject:
        def __init__(self, results):
            self.results = results

    results = dict()
    sets    = []
    data = kwargs['dataset']

    for i in range(0, len(data)):
        results.update({data.iloc[i]['method'] + '_' + str(data.iloc[i]['id']) : [float(data.iloc[i]['score'])]})

    return AnalysisObject(results)

def plot(**kwargs):

    absolute        = kwargs['absolute'] if 'absolute' in kwargs else False
    analysis_object = kwargs['analysis_object']
        
    plot = None
    
    if kwargs['plot_type'].lower() == 'stacked':
        plot = Stacked(analysis_object.results)
    if kwargs['plot_type'].lower() == 'enrichment':
        plot = Enrichment(analysis_object.results, absolute)
        
    plot.run(kwargs['output_filename'])

class AnalysisFactory:

    def __init__(self, dataset, target_name, transforms, models, metric):

        self.dataset = dataset
        self.dataset_shape = dataset.shape
        self.model_transform_tuples = list(product(transforms, models))
        self.parameters  = {'target_name' : target_name, 'metric' : metric, 'num_iterations' : 5, 'num_folds' : 5}

        self.analysis_runs = []

        for transform, model in list(product(transforms, models)):

            analysis_transform = None
            analysis_model = None

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

            self.analysis_runs.append(ModelTransformAnalysis(analysis_transform, analysis_model, self.parameters))

    def run(self):

        self.results = dict()
        
        for i, analysis in enumerate(self.analysis_runs):
            result_key = '_'.join(analysis.transform_instance.get_name().lower().split(' ')) + '_' + '_'.join(analysis.model_instance.get_name().lower().split(' ')) 
            print('Analyzing: ' + result_key)
            self.results[result_key] = analysis.get_performance(self.dataset)

        return self
