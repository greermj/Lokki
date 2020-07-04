from lokki.lib import PipelineComponents


class ModelSelection:

    def __init__(self, dataset, taxonomy, mode, k, analysis_object):

        self.dataset = dataset
        self.dataset_shape = dataset.shape
        self.taxonomy = taxonomy 
        self.mode = mode
        self.k = k
        self.analysis_object = analysis_object


        self.results         = sorted(analysis_object.results, 
                                      key = lambda x : x['value'], 
                                      reverse = True)

        print(self.results[0])
        print(self.mode)

    def train(self):
        pass

    def predict(self):
        pass
