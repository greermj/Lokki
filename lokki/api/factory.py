from itertools import product 

def configure(**kwargs):
    return AnalysisFactory(kwargs['dataset'], kwargs['target_name'], kwargs['models'], kwargs['transforms'], kwargs['metric'])

class AnalysisFactory:
    def __init__(self, dataset, target_name, models, transforms, metric):

        self.dataset = dataset
        self.model_transform_tuples = list(product(models, transforms))
        self.target_name = target_name
        self.metric = metric 

    def run(self):
        pass
    
