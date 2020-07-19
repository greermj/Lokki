from lokki.lib import PipelineComponents
import sys


class ModelSelection:

    def __init__(self, kwargs):

        self.dataset         = kwargs['dataset'] if 'dataset' in kwargs else sys.exit('ERROR: Could not find dataset to perform model selection')
        self.taxonomy        = kwargs['taxonomy'] if 'taxonomy' in kwargs else sys.exit('ERROR: Could not find taxonomy to perform model selection')
        self.mode            = kwargs['mode'].lower() if 'mode' in kwargs else 'optimal'
        self.k               = kwargs['k'] if 'k' in kwargs else 3
        self.analysis_object = kwargs['analysis_object'] if 'analysis_object' in kwargs else sys.exit('ERROR: Could not find analysis object to perform model selection')
        self.parameters      = self.analysis_object.results[0]['parameters']
        self.dataset_shape = self.dataset.shape
        self.pipeline_components = PipelineComponents(self.dataset_shape, self.parameters, self.taxonomy)

        self.results = sorted(self.analysis_object.results, 
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
                self.grid = self.results[0]['grid']

            else:
                pipeline_build = (robust_data_transform, robust_feature_transform, robust_model)
                self.grid = self._get_robust_grid(pipeline_build)

        else:
            # Return the highest scoring pipeline
            pipeline_build = self.results[0]['key']
            self.grid = self.results[0]['grid']

        # Retrieve components 
        self.analysis_data_transform    = self.pipeline_components.get_component( pipeline_build[0].lower(),    'data_transform')
        self.analysis_feature_transform = self.pipeline_components.get_component( pipeline_build[1].lower(), 'feature_transform')
        self.analysis_model             = self.pipeline_components.get_component( pipeline_build[2].lower(),             'model')

        self.pipeline_build = pipeline_build
        print('Selected Pipeline: ' + str(pipeline_build))

        # Split into data and targets 
        X = self.dataset.loc[:, [x.lower().startswith('otu') for x in self.dataset.columns.values]].copy().reset_index(drop = True)
        y = self.dataset.loc[:, [x.lower().startswith('target') for x in self.dataset.columns.values]].copy().reset_index(drop = True).iloc[:,0].values

        # Apply any data transforms 
        self.analysis_data_transform.fit(X, y)
        X_train = self.analysis_data_transform.transform(X, y)

        # Apply any feature transforms 
        self.analysis_feature_transform.fit(self.grid, X_train, y)
        X_train = self.analysis_feature_transform.transform(X_train, y)

        # Train the model 
        self.analysis_model.fit(X_train, y)

    # Description: Apply data transform, feature selection or engineering, and model.predict()
    def predict(self, data):
        X = data.loc[:, [x.lower().startswith('otu')    for x in data.columns.values]].copy().reset_index(drop = True).copy()
        data_transformed_X    = self.analysis_data_transform.transform(X).copy()
        feature_transformed_X = self.analysis_feature_transform.transform(data_transformed_X).copy()
        return self.analysis_model.predict(feature_transformed_X)

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
