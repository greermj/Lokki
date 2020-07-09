from lokki.lib import PipelineComponents
import sys


class ModelSelection:

    def __init__(self, dataset, taxonomy, mode, k, analysis_object):

        self.dataset = dataset
        self.dataset_shape = dataset.shape
        self.taxonomy = taxonomy 
        self.mode = mode
        self.k = k
        self.analysis_object = analysis_object
        self.pipeline_components = PipelineComponents(self.dataset_shape, self.taxonomy)

        self.results = sorted(analysis_object.results, 
                              key = lambda x : x['value'], 
                              reverse = True)

        if self.results == []:
            return

        if self.mode.lower() == 'robust':

            robust_data_transform = None
            robust_feature_transform = None
            robust_model = None

            data_transform_counts = dict()
            feature_transform_counts = dict()
            model_counts = dict()

            # For each sorted result dictionary, extract the pipeline components and maintain a count. Once k is hit for 
            # each component store and return the components that hit the target count (k) first
            for elem in self.results:
                data_transform, feature_transform, model = elem['key']

                data_transform_counts = self._update_count(data_transform, data_transform_counts)
                feature_transform_counts = self._update_count(feature_transform, feature_transform_counts)
                model_counts = self._update_count(model, model_counts)

                if robust_data_transform == None:
                    for x, count in data_transform_counts.items():
                        if count == self.k:
                            robust_data_transform = x

                if robust_feature_transform == None:
                    for x, count in feature_transform_counts.items():
                        if count == self.k:
                            robust_feature_transform = x

                if robust_model == None:
                    for x, count in model_counts.items():
                        if count == self.k:
                            robust_model = x

            pipeline_build = (robust_data_transform, robust_feature_transform, robust_model)

        else:

            # Return the highest scoring pipeline
            pipeline_build = self.results[0]['key']

        # Retrieve components and train model 
        analysis_data_transform    = self.pipeline_components.get_component( pipeline_build[0].lower(),    'data_transform')
        analysis_feature_transform = self.pipeline_components.get_component( pipeline_build[1].lower(), 'feature_transform')
        analysis_model             = self.pipeline_components.get_component( pipeline_build[2].lower(),             'model')

        print(analysis_data_transform)
        print(analysis_feature_transform)
        print(analysis_model)

    def _update_count(self, elem, dict_counts):
        if not elem in dict_counts:
            dict_counts[elem] = 1
        else:
            dict_counts[elem] += 1
        return dict_counts 

    def predict(self):
        pass
