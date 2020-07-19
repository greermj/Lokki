import sys
import os
import lokki

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from feather import read_dataframe 

def get_score(pred, y, metric = 'auc'):
    score = 0
    if metric == 'auc':
        score = roc_auc_score(np.array(y).astype(bool), np.array(pred).astype(bool))
    elif metric == 'precision':
        score = precision_score(np.array(y).astype(bool), np.array(pred).astype(bool))
    elif metric == 'recall':
        score = recall_score(np.array(y).astype(bool), np.array(pred).astype(bool))
    return score

dataset = 'zeller' #sys.argv[1]
path_to_feathers = '/Users/michael/Documents/Labs/Zhang/Lokki/dev/pipeline_test_data/feathers/'#sys.argv[2]
path_to_data = '/Users/michael/Documents/Labs/Zhang/Lokki/dev/pipeline_test_data/all/'#sys.argv[3]
path_to_taxonomy = './pipeline_study/taxonomy/'#sys.argv[4]
path_to_output = './out/'#sys.argv[5]
pheno_name_one = 'control'#sys.argv[6]
pheno_name_two = 'cancer'#sys.argv[7]

if not os.path.exists(path_to_output + '/' + dataset):
    os.makedirs(path_to_output + '/' + dataset)

otu  = read_dataframe(path_to_feathers + '/' + dataset + '.0.03.otu.feather')
meta = pd.read_csv(path_to_data + '/' + dataset + '.metadata', sep='\t')
tax  = pd.read_csv(path_to_taxonomy + '/' + dataset + '.taxonomy', sep = '\t')
missing_threshold = 0.80

# The baxter metadata file does not include disease classifications for every sample so I needed to only include samples
# for which we have a disease classification
num_folds = 5
mult = 13
max_k = 10
cv_results = pd.DataFrame()

for fold in range(num_folds):
    fold_results = dict()
    fold_results['dataset'] = dataset 
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
                                      data_transforms = ['none', 'log', 'zscore'],#['none', 'log', 'zscore'],
                                      feature_transforms = ['none', 'pca'],# 'hfe', 'chi_square', 'mutual_information', 'factor_analysis', 'ica'],#['none', 'pca', 'hfe', 'chi_square', 'mutual_information', 'factor_analysis', 'ica', 'nmf'],
                                      models = ['decision_tree', 'extra_tree'],# 'adaboost', 'gradient_boosting', 'svm', 'ridge_regression'],#['random_forest', 'decision_tree', 'lda', 'qda', 'extra_tree', 'logistic_regression', 'adaboost', 'gradient_boosting', 'svm', 'ridge_regression'],
                                      metric = 'auc',
                                      taxonomy = tax)

    analysis_object = analysis_config.run()

    optimal = lokki.select(dataset = X_train,
                           taxonomy = tax,
                           analysis_object = analysis_object)

    optimal_train_preds = optimal.predict(X_train)
    optimal_test_preds  = optimal.predict(X_test)

    fold_results['optimal'] = get_score(optimal_test_preds, y_test)

    for i in range(1, max_k):
        robust = lokki.select(dataset = X_train,
                              taxonomy = tax,
                              mode = 'robust',
                              k = i,
                              analysis_object = analysis_object)

        robust_train_preds = robust.predict(X_train)
        robust_test_preds  = robust.predict(X_test)

        fold_results['robust_k_' + str(i)] = get_score(robust_test_preds, y_test)

    cv_results = cv_results.append(fold_results, ignore_index = True)
    cv_results.to_csv(path_to_output + '/' + dataset + '/' + dataset + '_cv_results.csv', index = False)
