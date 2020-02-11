import pandas as pd

import lokki

data = pd.read_csv('./wdbc.csv')

analysis = lokki.configure(dataset = data,
                           target_name = 'Target',
                           transforms = ['pca', 'none', 'chi_square'],
                           models = ['random_forest', 'svm', 'logistic_regression'],
                           metric = 'auc')

results = analysis.run()

results.visualize()
