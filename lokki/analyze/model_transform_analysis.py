# Notes: Should run data through each grid transform and then the model and return the top performance 
class ModelTransformAnalysis:
    
    def __init__(self, transform_instance, model_instance, parameters):

        self.transform_instance = transform_instance
        self.model_instance = model_instance
        self.parameters = parameters

    # For each param in grid run transform then get and store model performance (keep track of max)
    def get_performance(self, dataset):

        hyperparameter_grid = self.transform_instance.hyperparameter_grid()

        if hyperparameter_grid != None:

            X = dataset.drop(self.parameters['target_name'], axis = 1).copy()
            y = dataset[self.parameters['target_name']].copy()

            optimal_score = None
            
            for grid in self.transform_instance.hyperparameter_grid():

                X_train = self.transform_instance.fit_transform(grid, X, y)
                

        else:

            pass
