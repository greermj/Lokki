import pandas as pd
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt

import lokki

#data = pd.read_csv('./dev/new_test_data.csv')
#results = lokki.custom(dataset = data,
#                       scoring_metric_name = 'AUROC')

#results = pickle.load(open('./dev/microbiome_results/ahn_ctrl_tmr.p', 'rb'))
results = pickle.load(open('./pipeline_study/data/baxter/all/baxter_tumor_results.p', 'rb'))

lokki.plot(analysis_object = results,
           plot_type = 'performance',
           filename = 'distribution.png')
