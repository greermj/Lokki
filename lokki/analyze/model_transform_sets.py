class ModelTransformSets:

    def __init__(self, results):
        self.results = results
    
    def get_model_transform_sets(self):

        enrichment_sets = ['factor', 'ica', 'nmf', 'pca', 'none', 
                           'chi_square', 'mutual_information', 'random_forest', 'decision_tree', 
                           'extra_tree', 'lda', 'qda', 'logistic_regression', 'ridge', 
                           'adaboost', 'gradient_boosting', 'svm']

        aggregate_sets  = {'tree-based' : ['decision_tree', 'extra_tree'], 
                           'ensemble' : ['random_forest', 'adaboost', 'gradient_boosting'], 'linear' : ['svm', 'ridge'],
                           'feature_selection' : ['mutual_information', 'none', 'chi_square'], 'feature_engineering' : ['pca', 'ica', 'factor', 'nmf']}

        return enrichment_sets, aggregate_sets
