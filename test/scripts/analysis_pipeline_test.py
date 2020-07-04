import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle

import sys
import lokki

#filename = sys.argv[1]

data = pd.read_csv('./dev/baxter/sample_data_baxter_tumor.csv')
taxonomy = pd.read_csv('./dev/baxter/baxter.taxonomy', sep='\t')

# If less than 70% are missing or its the target column add the column (eg 55% missing = ok, 65% missing = ok, 71% missing = not ok)
data = data.loc[:, ((data == 0).mean() < 0.7) | (data.columns.values == 'target')].copy()

analysis = lokki.configure(dataset = data,
                           target_name = 'target',
                           data_transforms = ['none'],# 'log', 'zscore'],
                           feature_transforms = ['none', 'chi_square'],#'hfe', 'chi_square', 'mutual_information', 'factor', 'ica', 'nmf'],
                           models = ['random_forest', 'decision_tree'],# 'lda', 'qda', 'extra_tree', 'logistic_regression', 'adaboost', 'gradient_boosting', 'svm', 'ridge'],
                           metric = 'auc',
                           taxonomy = taxonomy)

results = analysis.run()

pickle.dump(results, open('neo.p', 'wb'))
