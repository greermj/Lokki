import sys

import numpy as np

import matplotlib.pyplot as plt

class Enrichment:

    def __init__(self, results):

        self.results = results 

        self.enrichment_sets = ['factor', 'ica', 'nmf', 'pca', 'none', 'chi_square', 'mutual_information',
                                'random_forest', 'decision_tree', 'extra_tree', 'lda', 'qda', 'logistic_regression', 'ridge', 'adaboost', 'gradient_boosting', 'svm']

        self.aggregate_sets  = {'tree-based' : ['decision_tree', 'extra_tree'], 'ensemble' : ['random_forest', 'adaboost', 'gradient_boosting'], 'linear' : ['svm', 'ridge'],
                                'feature_selection' : ['mutual_information', 'none', 'chi_square'], 'feature_engineering' : ['pca', 'ica', 'factor', 'nmf']}
                                
    def run(self, filename):

        for i, eset in enumerate(self.enrichment_sets):

            current_sets = [x for x in self.results.keys() if eset in x]

            plt.figure(figsize=(28, 2), dpi=100)
            plt.xlim(1, 0)
            plt.xticks([]),plt.yticks([])
            plt.ylabel(eset, fontsize=15)
            print(i)
            print(eset)
            print('\n\n')
            for transform_model in current_sets:
                plt.axvline(x=np.mean(self.results[transform_model]), color='k', linewidth = 4)

            plt.savefig('./dev/' + eset + '.png', dpi=100)

    def get_ranked_list(self):
        return list({k: v for k, v in sorted(self.results.items(), key=lambda item: np.mean(item[1]), reverse = True)}.keys())
