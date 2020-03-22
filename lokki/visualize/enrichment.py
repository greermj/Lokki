import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from lokki.analyze import ModelTransformSets

class Enrichment:

    def __init__(self, analysis_object):
        self.analysis_object = analysis_object

    def run(self, filename):

        enrichment_plot_data = []
        sorted_results = self.get_ranked_list()
        dimensions = list(set([x for y in self.analysis_object.results for x in y['key']]))

        # Add single dimension data
        for dimension in dimensions:
            enrichment_results = []
            dimension_results = [x for x in self.analysis_object.results if dimension in x['key']]

            # Establish map between key and rank in sorted list (ie enumerate data)
            for x in dimension_results:
                for i, y in enumerate(sorted_results):
                    if x['key'] == y['key']:
                       enrichment_results.append({'key': x['key'], 'rank' : i}) 

            enrichment_plot_data.append(enrichment_results)
            print(dimension)
            print(dimension_results)
            print('\n')

        # Add combination dimension data


        print(sorted_results)

        sorted_len = len(sorted_results)
        linewidth = (100 / sorted_len) * 3.5
        plt.xlim(0 - 1/(linewidth/2), 1 + 1/(linewidth/2))
 
        for plot in enrichment_plot_data:
    
            for x in plot:
                plt.axvline( ( 1/ sorted_len) * x['rank'], linewidth = (100 / sorted_len) * 3.5)
            print('plot')
            print(plot)
            break



        return sorted_results, enrichment_plot_data, self.analysis_object


         

        '''
        print(self.get_ranked_list())

        for i, eset in enumerate(self.custom_sets):
            print(eset)

            plt.figure(figsize=(28, 2), dpi=100)
            plt.xlim(0, len(self.get_ranked_list()) * self.unit)
            plt.xticks([]),plt.yticks([])
            plt.ylabel(eset, fontsize=25)


            in_group = []
            out_group = []

            print('\n')
            for i, current_model in enumerate(self.get_ranked_list()):

                if current_model.lower().startswith(eset.lower()):
                    plt.axvline(x= i * self.unit + self.offset, color='k', linewidth = self.linewidth)
                    in_group.append(self.results[current_model][0])
                else:
                    out_group.append(self.results[current_model][0])

            ks_stat, pvalue = stats.ks_2samp(in_group, out_group)

            print('ks stat: ' + str(ks_stat) + '\tp-value: ' + str(pvalue))

            plt.savefig('./dev/' + eset + '.png', dpi=100)
            '''

        print('done')

    def get_ranked_list(self):
        return sorted(self.analysis_object.results, key = lambda x : x['value'], reverse = True)
