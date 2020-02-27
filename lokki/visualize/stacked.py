import numpy as np

import matplotlib.pyplot as plt

class Stacked:

    def __init__(self, results):
        self.results = results 

    def run(self, filename):


        keys = tuple(self.results.keys())
        y_pos = np.arange(len(keys))

        scores = []
        labels = []
        all_results = dict()
        
        for label in keys:
            scores.append(np.mean(self.results[label]))
            labels.append(label)
            all_results[label] = np.mean(self.results[label])

        sorted_results = {k: v for k, v in sorted(all_results.items(), key=lambda item: item[1], reverse = True)}

        sorted_keys = tuple(sorted_results.keys())
        sorted_values = tuple(sorted_results.values())

        plt.figure(figsize=(40, 100), dpi=100)
        plt.bar(y_pos, sorted_values)
        plt.xticks(y_pos, sorted_keys, rotation = 75, fontsize = 45)
        plt.yticks(fontsize = 60)
        plt.ylabel('AUC',fontsize=60, labelpad=80)
        plt.savefig(filename, dpi=100)
