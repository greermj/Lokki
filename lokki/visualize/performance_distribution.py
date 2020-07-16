import os

import numpy as np

import matplotlib.pyplot as plt

class PerformanceDistribution:

    def __init__(self, analysis_object, kwargs):
        self.analysis_object = analysis_object
        self.kwargs = kwargs 
        self.output_filename = kwargs['filename'] if 'filename' in kwargs else 'performance_distribution.png'

        # Creates output directory to store performance distributions (note: I create separate run folders so the user wont overwrite existing figures)
        count = 0
        while True:
            if not os.path.exists('./performance_figs/run_' + str(count) + '/'):
                os.makedirs('./performance_figs/run_' + str(count))
                self.output_directory = './performance_figs/run_' + str(count) + '/'
                break
            else:
                count += 1

    def run(self):

        data = { '_'.join(x['key']) : x['value'] for x in self.analysis_object.results}
        y_pos = np.arange(len(data))

        sorted_results = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse = True)}
        sorted_keys = tuple(sorted_results.keys())
        sorted_values = tuple(sorted_results.values())

        # Output legend
        with open(self.output_directory + '/legend.txt', 'w') as writer:
            writer.write('rank\tdescription\tscore\n')

            for i in y_pos:
                writer.write(str(i + 1) + '\t' + sorted_keys[i] + '\t' + str(sorted_values[i]) + '\n')
        
        # Output figure 
        plt.figure(figsize=(10,10), dpi=80)
        plt.style.use('seaborn-darkgrid')
        plt.bar(y_pos + 1, sorted_values, width = 1.0)
        plt.title('Performance Distribution', fontweight = 'bold', fontsize=18)
        plt.xticks([])
        plt.yticks(np.arange(0, max(sorted_values) + 0.05, 0.05))
        plt.ylabel(self.analysis_object.scoring_metric_name.upper(), fontweight = 'bold', fontsize=15)
        plt.savefig(self.output_directory + self.output_filename)
        return plt

    def tri_color(self, filename):
        pass
