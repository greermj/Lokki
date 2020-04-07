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

        # Output at most 40 bars in the stacked plot
        num_bars = 40
        if len(sorted_results) > num_bars:
            y_pos = np.arange(num_bars)
            sorted_keys = sorted_keys[:num_bars]
            sorted_values = sorted_values[:num_bars]

        # Output key
        with open('key.txt', 'w') as writer:
            for i in y_pos:
                writer.write(str(i) + ':\t' + sorted_keys[i] + '\n')
        
        # Output figure 
        plt.figure(figsize=(10,10), dpi=80)
        plt.style.use('seaborn-darkgrid')
        plt.bar(y_pos, sorted_values)
        plt.title('Performance Distribution')
        plt.xticks(y_pos)
        plt.ylabel(self.analysis_object.scoring_metric_name)
        plt.savefig(filename)
        return plt

    def tri_color(self, filename):
        pass
