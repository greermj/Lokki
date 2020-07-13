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

        grid = None

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

            if (robust_data_transform == None) or (robust_feature_transform == None) or (robust_model == None):
                print('WARNING: Could not find robust pipeline components, selecting optimal choice. Try reducing the parameter k')
                pipeline_build = self.results[0]['key']
                grid = self.results[0]['grid']

            else:
                pipeline_build = (robust_data_transform, robust_feature_transform, robust_model)
                grid = self._get_robust_grid(pipeline_build)

        else:
            # Return the highest scoring pipeline
            pipeline_build = self.results[0]['key']
            grid = self.results[0]['grid']

        # Retrieve components 
        self.analysis_data_transform    = self.pipeline_components.get_component( pipeline_build[0].lower(),    'data_transform')
        self.analysis_feature_transform = self.pipeline_components.get_component( pipeline_build[1].lower(), 'feature_transform')
        self.analysis_model             = self.pipeline_components.get_component( pipeline_build[2].lower(),             'model')

        # Retrieve default hyperparameter grid
        self.hyperparameter_grid = self.analysis_feature_transform.hyperparameter_grid()

        # Split into data and targets 
        X = self.dataset.loc[:, [x.lower().startswith('otu') for x in self.dataset.columns.values]].copy().reset_index(drop = True)
        y = self.dataset.loc[:, [x.lower().startswith('target') for x in self.dataset.columns.values]].copy().reset_index(drop = True).iloc[:,0].values

        # Apply any data transforms 
        self.analysis_data_transform.fit(X, y)
        X_train = self.analysis_data_transform.transform(X, y)

        # Apply any feature transforms 
        if grid == None:
            self.analysis_feature_transform.fit(X_train, y)
            X_train = self.analysis_feature_transform.transform(X_train, y)
        else:
            self.analysis_feature_transform.fit(grid, X_train, y)
            X_train = self.analysis_feature_transform.transform(X_train, y)

        # Train the model 

        
    def predict(self):
        pass

    # Description: Updates the count of the number of times the element was hit in the dictionary of counts
    def _update_count(self, elem, dict_counts):
        if not elem in dict_counts:
            dict_counts[elem] = 1
        else:
            dict_counts[elem] += 1
        return dict_counts 

    # Description: Finds the specific results data for the pipeline build (ie combination of data transform, feature transform and model) then returns the grid
    def _get_robust_grid(self, pipeline_build):
        for elem in self.results:
            if pipeline_build == elem['key']:
                return elem['grid']
        print('WARNING: Could not find robust pipeline grid, selecting optimal choice')
        return self.results[0]['grid']
