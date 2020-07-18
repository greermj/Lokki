import warnings
warnings.filterwarnings('ignore')

from itertools import product 

from lokki.analyze import ModelTransformAnalysis 
from lokki.analyze import AnalysisObject

from lokki.lib import PipelineComponents 
from lokki.selection import ModelSelection 

# Visualizations 
from lokki.visualize import PerformanceDistribution
from lokki.visualize import Enrichment 

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

def select(**kwargs):
    return ModelSelection(kwargs)

def plot(**kwargs):

    analysis_object = kwargs['analysis_object']
        
    plot = None
    
    if kwargs['plot_type'].lower() == 'performance':
        plot = PerformanceDistribution(analysis_object, kwargs)
    if kwargs['plot_type'].lower() == 'enrichment':
        plot = Enrichment(analysis_object, kwargs)

    return plot.run()

class AnalysisFactory:

    def __init__(self, dataset, target_name, data_transforms, feature_transforms, models, metric, taxonomy):

        self.dataset = dataset
        self.dataset_shape = dataset.shape
        self.taxonomy = taxonomy
        self.model_transform_tuples = list(product(feature_transforms, models))
        self.parameters  = {'target_name' : target_name, 'metric' : metric, 'num_iterations' : 5, 'num_folds' : 3}
        self.pipeline_components = PipelineComponents(self.dataset_shape, self.parameters, self.taxonomy)

        self.analysis_runs = []

        # For each combination of data_tranform, feature_transform, and model
        for data_transform, feature_transform, model in list(product(data_transforms, feature_transforms, models)):

            # Retrieve the component object 
            analysis_data_transform    = self.pipeline_components.get_component( data_transform.lower(),    'data_transform') 
            analysis_feature_transform = self.pipeline_components.get_component( feature_transform.lower(), 'feature_transform')
            analysis_model             = self.pipeline_components.get_component( model.lower(),             'model')

            # Append an model transform analysis object that will be run at a later point 
            self.analysis_runs.append( ModelTransformAnalysis(analysis_data_transform, 
                                                              analysis_feature_transform, 
                                                              analysis_model, 
                                                              self.parameters))

    def run(self):

        self.results = []
        
        for i, analysis in enumerate(self.analysis_runs):

            current_data_transform    = '_'.join(analysis.data_transform_instance.get_name().lower().split(' '))
            current_feature_transform = '_'.join(analysis.feature_transform_instance.get_name().lower().split(' '))
            current_model             = '_'.join(analysis.model_instance.get_name().lower().split(' '))

            print('Analyzing: ' + current_data_transform + '_' + current_feature_transform + '_' + current_model)

            # The order of the key is important and I assume this order in components.py
            self.results.append({'key'          : (current_data_transform.strip().lower(), 
                                                   current_feature_transform.strip().lower(), 
                                                   current_model.strip().lower()), 
                                 'value'        : analysis.get_performance(self.dataset),
                                 'grid'         : analysis.grid,
                                 'parameters'   : analysis.parameters})

        return AnalysisObject(self.results, self.parameters['metric'])
