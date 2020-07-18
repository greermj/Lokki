import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle

import sys
import lokki

dataset = 'baxter'
path_to_dataset  = './dev/baxter/sample_data_baxter_tumor.csv'
path_to_taxonomy = './dev/baxter/baxter.taxonomy'

data = pd.read_csv(path_to_dataset)
data = data.loc[:, ((data == 0).mean() < 0.7) | np.array([x.lower().startswith('target') for x in data.columns.values])].copy()

taxonomy = pd.read_csv(path_to_taxonomy, sep='\t')

analysis = lokki.configure(dataset = data,
                           target_name = 'target',
                           data_transforms = ['zscore'],#['none', 'log', 'zscore'],
                           feature_transforms = ['none', 'chi_square', 'pca', 'ica', 'nmf', 'factor_analysis', 'mutual_information', 'hfe'],#['none', 'pca', 'hfe', 'chi_square', 'mutual_information', 'factor_analysis', 'ica', 'nmf'],
                           models = ['decision_tree'],#['random_forest', 'decision_tree', 'lda', 'qda', 'extra_tree', 'logistic_regression', 'adaboost', 'gradient_boosting', 'svm', 'ridge'],
                           metric = 'auc',
                           taxonomy = taxonomy)

results = analysis.run()

pickle.dump(results, open(dataset + '_tumor_results.p', 'wb'))
