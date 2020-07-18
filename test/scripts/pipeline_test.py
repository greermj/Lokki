import lokki

import pandas as pd
import dill as pickle 
import numpy as np
import matplotlib.pyplot as plt

path_to_train_data          = './pipeline_study/data/baxter/ahn/baxter_tumor.csv'
path_to_train_data_results  = './pipeline_study/data/baxter/ahn/baxter_tumor_results.p'
path_to_train_data_taxonomy = './pipeline_study/taxonomy/baxter.taxonomy'

path_to_test_data = './pipeline_study/data/baxter/ahn/ahn_tumor.csv'

# Load data
train_data  = pd.read_csv(path_to_train_data)
test_data   = pd.read_csv(path_to_test_data)

train_data_taxonomy = pd.read_csv(path_to_train_data_taxonomy, sep = '\t')

# Zero filtering 
train_data = train_data.loc[:, ((train_data == 0).mean() < 0.7) | (train_data.columns.values == 'target')].copy()
test_data  = test_data.loc[:,  ((test_data  == 0).mean() < 0.7) | (test_data.columns.values  == 'target')].copy()


# Load results from running search 
analysis_object = pickle.load(open(path_to_train_data_results, 'rb'))
'''
a=lokki.select(dataset = data,
             taxonomy = taxonomy,
             mode = 'robust',
             k = 2,
             analysis_object = analysis_object)
'''
