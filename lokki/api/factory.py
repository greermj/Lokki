from itertools import product 

from lokki.analyze import ModelTransformAnalysis 

# Models 
from lokki.model import RandomForest
from lokki.model import LogisticRegressionModel
from lokki.model import SVM

# Tranforms 
from lokki.transform import PCA
from lokki.transform import ChiSquare
from lokki.transform import Void

# Visualizations 
from lokki.visualize import Stacked 

def configure(**kwargs):
    return AnalysisFactory(kwargs['dataset'], kwargs['target_name'], kwargs['transforms'], kwargs['models'], kwargs['metric'])

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

            if transform.lower() == 'none':
                analysis_transform = Void(self.dataset_shape)
            elif transform.lower() == 'pca':
                analysis_transform = PCA(self.dataset_shape)
            elif transform.lower() == 'chi_square':
                analysis_transform = ChiSquare(self.dataset_shape)
            else:
                print('Error: Transform method not found')

            if model.lower() == 'random_forest':
                analysis_model = RandomForest()
            elif model.lower() == 'logistic_regression':
                analysis_model = LogisticRegressionModel()
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


    def visualize(self):
        Stacked(self.results)
