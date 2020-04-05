import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Notes: Should run data through each grid transform and then the model and return the top performance 
class ModelTransformAnalysis:
    
    def __init__(self, process_instance, transform_instance, model_instance, parameters):
    
        self.process_instance = process_instance
        self.transform_instance = transform_instance
        self.model_instance = model_instance
        self.parameters = parameters

    def get_performance(self, dataset):

        # The dataset 
        X = dataset.drop(self.parameters['target_name'], axis = 1).copy().reset_index(drop = True)
        y = dataset[self.parameters['target_name']].copy().reset_index(drop = True)

        # Final result returned; it is a list of means over the CV folds 
        iteration_performance_results = []

        # Transform hyperparameter options
        hyperparameter_grid = self.transform_instance.hyperparameter_grid()

        # For each iteration of the nested CV
        for i, iteration in enumerate(range(self.parameters['num_iterations'])):

            # If this analysis has a transform step 
            if hyperparameter_grid != None:

                optimal_score = 0

                # Determine which transform hyperparameter has the highest OOS performance using CV 
                for j, grid in enumerate(self.transform_instance.hyperparameter_grid()):

                    skf = StratifiedKFold(random_state = 2*i + j + 1 , n_splits = self.parameters['num_folds'])

                    fold_results = []
                
                    # For each fold
                    for train_index, test_index in skf.split(X, y):
            
                        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                        y_train, y_test = y[train_index], y[test_index]

                        # Preprocessing step 
                        self.process_instance.fit(X_train, y_train)
                        X_train = self.process_instance.transform(X_train, y_train)
                        X_test  = self.process_instance.transform(X_test, y_test)

                        # Prepare training data
                        self.transform_instance.fit(grid, X_train, y_train)

                        transformed_X_train = self.transform_instance.transform(X_train, y_train)
                        transformed_X_test = self.transform_instance.transform(X_test, y_test)

                        # Get performance 
                        performance = self.model_instance.evaluate(self.parameters, transformed_X_train, transformed_X_test, y_train, y_test)

                        fold_results.append(performance)
        
                    if np.mean(fold_results) > optimal_score:
                        optimal_score = np.mean(fold_results)

                # Store the mean 
                iteration_performance_results.append(optimal_score)

            else:

                # Perform same analysis above only without grid searching of transform hyperparameters 
                skf = StratifiedKFold(random_state = i, n_splits = self.parameters['num_folds'])

                fold_results = []

                optimal_score = 0

                for train_index, test_index in skf.split(X, y):

                    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                    y_train, y_test = y[train_index], y[test_index]
                        
                    # Preprocessing step 
                    self.process_instance.fit(X_train, y_train)
                    X_train = self.process_instance.transform(X_train, y_train)
                    X_test  = self.process_instance.transform(X_test, y_test)

                    # Prepare training data
                    self.transform_instance.fit('', X_train, y_train)

                    transformed_X_train = self.transform_instance.transform(X_train, y_train)
                    transformed_X_test = self.transform_instance.transform(X_test, y_test)

                    performance = self.model_instance.evaluate(self.parameters, transformed_X_train, transformed_X_test, y_train, y_test)

                    fold_results.append(performance)
    
                if np.mean(fold_results) > optimal_score:
                    optimal_score = np.mean(fold_results)
                
                iteration_performance_results.append(optimal_score)
        
        return np.mean(iteration_performance_results)
