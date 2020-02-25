import pandas as pd

import lokki

# Only for testing
import dill as pickle
#
''''
data = pd.read_csv('./dev/wdbc.csv')

analysis = lokki.configure(dataset = data,
                           target_name = 'Target',
                           transforms = ['pca', 'mutual_information', 'ica', 'factor', 'nmf', 'chi_square', 'none'],
                           models = ['random_forest', 'decision_tree', 'extra_tree', 'ridge', 'adaboost', 'svm', 'gradient_boosting', 'lda', 'qda', 'logistic_regression'],
                           metric = 'auc')

results = analysis.run()

pickle.dump(results, open('results.p', 'wb'))
'''
results = pickle.load(open('results.p', 'rb'))

lokki.plot(analysis_object = results,
           plot_type = 'stacked',
           #plot_type = 'enrichment',
           output_filename = 'out.png')
