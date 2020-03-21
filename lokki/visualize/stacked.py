import numpy as np

import matplotlib.pyplot as plt

class Stacked:

    def __init__(self, analysis_object):
        self.analysis_object = analysis_object

    def run(self, filename):

        data = { '_'.join(x['key']) : x['value'] for x in self.analysis_object.results}
        y_pos = np.arange(len(data))

        sorted_results = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse = True)}
        sorted_keys = tuple(sorted_results.keys())
        sorted_values = tuple(sorted_results.values())

        plt.figure(figsize=(80, 100))
        plt.bar(y_pos, sorted_values)
        plt.xticks(y_pos, sorted_keys, rotation = 75, fontsize = 45)
        plt.yticks(fontsize = 70)
        plt.ylabel(self.analysis_object.scoring_metric_name,fontsize=100, labelpad=80)
        plt.savefig(filename, dpi=100)
