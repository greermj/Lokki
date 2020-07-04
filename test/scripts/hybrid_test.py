import pandas as pd
import numpy as np
import dill as pickle

import matplotlib.pyplot as plt

import lokki

data = pd.read_csv('./dev/new_test_data.csv')

results = lokki.custom(dataset = data,
                       scoring_metric_name = 'wpc-index')

#results = pickle.load(open('./dev/microbiome_results/hale_ctrl_tmr.p', 'rb'))

lokki.plot(analysis_object = results,
           plot_type = 'hybrid',
           min_hits = 2,
           output = 'u.png')

