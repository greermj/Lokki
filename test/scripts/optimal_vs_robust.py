import sys

import pandas as pd

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

# The baxter metadata file does not include disease classifications for every sample so I needed to only include samples
# for which we have a disease classification
fold = 0
#for fold in range(num_folds):
control_tumor_samples   = [x for x in otu['Group'] if meta[meta['sample'] == x]['disease'].values == pheno_name_one or meta[meta['sample'] == x]['disease'].values == pheno_name_two]
control_tumor_mask   = [True if x in control_tumor_samples else False for x in otu['Group']]
otu_control_tumor   = otu.loc[control_tumor_mask, :]
otu_control_tumor= otu_control_tumor.loc[:, [x for x in otu_control_tumor.columns.values if x.lower().startswith('group') or x.lower().startswith('otu')]].copy()
otu_control_tumor.rename(columns={'Group' : 'Sample'}, inplace = True)
control_tumor_targets   = [1 if meta[meta['sample'] == x]['disease'].values == pheno_name_two else 0 for x in otu_control_tumor['Sample']]
tmr_X_train, tmr_X_test, tmr_y_train, tmr_y_test = train_test_split(otu_control_tumor, control_tumor_targets, test_size=0.2, random_state=fold, stratify = control_tumor_targets)
tmr_X_train = tmr_X_train.reset_index(drop = True).copy()
tmr_X_test = tmr_X_test.reset_index(drop = True).copy()
tmr_X_train['Target'] = tmr_y_train
tmr_X_test['Target'] = tmr_y_test

#tmr_X_train.to_csv('./train_data_' + dataset + '_tumor.csv', index = False)
#tmr_X_test.to_csv('./test_data_' + dataset + '_tumor.csv', index = False)
