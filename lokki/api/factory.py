from itertools import product 

from lokki.analyze import ModelTransformAnalysis 

# Models 
from lokki.model import RandomForest
from lokki.model import LogisticRegression

# Tranforms 
from lokki.transform import PCA
from lokki.transform import ChiSquare
from lokki.transform import Void

def configure(**kwargs):
    return AnalysisFactory(kwargs['dataset'], kwargs['target_name'], kwargs['transforms'], kwargs['models'], kwargs['metric'])

class AnalysisFactory:

    def __init__(self, dataset, target_name, transforms, models, metric):

        self.dataset = dataset
        self.dataset_shape = dataset.shape
        self.model_transform_tuples = list(product(transforms, models))
        self.parameters  = {'target_name' : target_name, 'metric' : metric}

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
                print('bang')

            if model.lower() == 'random_forest':
                analysis_model = RandomForest()
            elif model.lower() == 'logistic_regression':
                analysis_model = LogisticRegression()
            else:
                print('hang')

            self.analysis_runs.append(ModelTransformAnalysis(analysis_transform, analysis_model, self.parameters))

    def run(self):

        result = dict()
        
        for i, analysis in enumerate(self.analysis_runs):
            print(analysis.transform_instance.get_model_name())
            result[i] = analysis.get_performance(self.dataset)
