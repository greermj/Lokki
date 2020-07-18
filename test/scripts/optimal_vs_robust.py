import sys
import lokki

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feather import read_dataframe 


#dataset = sys.argv[1]

path_to_feathers = '/Users/michael/Documents/Labs/Zhang/Lokki/dev/pipeline_test_data/feathers/'
path_to_data = '/Users/michael/Documents/Labs/Zhang/Lokki/dev/pipeline_test_data/all/'
path_to_taxonomy = './pipeline_study/taxonomy/'

dataset = 'ahn'
otu  = read_dataframe(path_to_feathers + '/' + dataset + '.0.03.otu.feather')
meta = pd.read_csv(path_to_data + '/' + dataset + '.metadata', sep='\t')
tax  = pd.read_csv(path_to_taxonomy + '/' + dataset + '.taxonomy', sep = '\t')

pheno_name_one = 'control'
pheno_name_two = 'cancer'

missing_threshold = 0.80

# The baxter metadata file does not include disease classifications for every sample so I needed to only include samples
# for which we have a disease classification
fold = 0
mult = 13
cv_results = pd.DataFrame()

#for fold in range(num_folds):
control_tumor_samples   = [x for x in otu['Group'] if meta[meta['sample'] == x]['disease'].values == pheno_name_one or meta[meta['sample'] == x]['disease'].values == pheno_name_two]
control_tumor_mask   = [True if x in control_tumor_samples else False for x in otu['Group']]
otu_control_tumor   = otu.loc[control_tumor_mask, :]
otu_control_tumor= otu_control_tumor.loc[:, [x for x in otu_control_tumor.columns.values if x.lower().startswith('group') or x.lower().startswith('otu')]].copy()
otu_control_tumor = otu_control_tumor.loc[:, ((otu_control_tumor == 0).mean() < missing_threshold) ].copy()
otu_control_tumor.rename(columns={'Group' : 'Sample'}, inplace = True)
control_tumor_targets   = [1 if meta[meta['sample'] == x]['disease'].values == pheno_name_two else 0 for x in otu_control_tumor['Sample']]
X_train, X_test, y_train, y_test = train_test_split(otu_control_tumor, control_tumor_targets, test_size=0.2, random_state=fold * mult, stratify = control_tumor_targets)
X_train = X_train.reset_index(drop = True).copy()
X_test = X_test.reset_index(drop = True).copy()
X_train['Target'] = y_train
X_test['Target'] = y_test

analysis_config = lokki.configure(dataset = X_train,
                                  target_name = 'target',
                                  data_transforms = ['none', 'log'],#['none', 'log', 'zscore'],
                                  feature_transforms = ['none', 'pca', 'chi_square', 'nmf'],#['none', 'pca', 'hfe', 'chi_square', 'mutual_information', 'factor_analysis', 'ica', 'nmf'],
                                  models = ['ridge_regression', 'svm', 'decision_tree', 'logistic_regression'],#['random_forest', 'decision_tree', 'lda', 'qda', 'extra_tree', 'logistic_regression', 'adaboost', 'gradient_boosting', 'svm', 'ridge_regression'],
                                  metric = 'auc',
                                  taxonomy = tax)

analysis_object = analysis_config.run()



optimal = lokki.select(dataset = X_train,
                       taxonomy = tax,
                       analysis_object = analysis_object)

robust = lokki.select(dataset = X_train,
                      taxonomy = tax,
                      mode = 'robust',
                      k = 2,
                      analysis_object = analysis_object)



#X_train.to_csv('./train_data_' + dataset + '_tumor.csv', index = False)
#X_test.to_csv('./test_data_' + dataset + '_tumor.csv', index = False)
