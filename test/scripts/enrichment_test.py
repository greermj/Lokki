import pandas as pd
import numpy as np
import dill as pickle

import matplotlib.pyplot as plt

import lokki

'''
data = pd.read_csv('./dev/new_test_data.csv')

results = lokki.custom(dataset = data,
                       scoring_metric_name = 'AUROC')
'''

#results = pickle.load(open('./dev/microbiome_results/ahn_ctrl_tmr.p', 'rb'))
results = pickle.load(open('./baxter_tumor_results.p', 'rb'))
#results = pickle.load(open('./dev/zeller_neos/zeller_adenoma_results.p', 'rb'))

lokki.plot(analysis_object = results,
           plot_type = 'enrichment',
           #filters = ['gradient_boosting'],
           mode = 'single',
           min_hits = 5,
           max_combn = 2,
           output = 'out_test.png')
