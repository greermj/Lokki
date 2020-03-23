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
        rank_to_value = dict()

        # Add single dimension data
        for dimension in dimensions:
            enrichment_results = []
            dimension_results = [x for x in self.analysis_object.results if dimension in x['key']]

            # Establish map between key and rank in sorted list (ie enumerate data)
            for x in dimension_results:
                for i, y in enumerate(sorted_results):
                    if x['key'] == y['key']:
                       enrichment_results.append({'key': x['key'], 'rank' : i, 'value' : x['value']}) 
                       rank_to_value[i] = x['value'] 

            enrichment_plot_data.append(enrichment_results)
            print(dimension)
            print(dimension_results)
            print('\n')

        # Add combination dimension data


        print(sorted_results)


        # Set layout parameters 
        sorted_len = len(sorted_results)
        linewidth = (100 / sorted_len) * 3.5
        num_plots_per_page = 10
        num_plots_per_page = num_plots_per_page if len(enrichment_plot_data) > num_plots_per_page else len(enrichment_plot_data)

        # Set up subplots
        current_plot_index = 0
        #plt.subplots_adjust(hspace = 0.4)
        fig, ax = plt.subplots(nrows = num_plots_per_page, ncols = 1)
        fig.tight_layout()
        #fig.set_figheight(5)
        for cur_ax in ax:
            cur_ax.set_xticks([])
            cur_ax.set_yticks([])

        # Plot data
        for i, plot in enumerate(enrichment_plot_data):

            # Reset subplots 
            if (i % num_plots_per_page == 0) and (i != 0):
                current_plot_index = 0
                fig, ax = plt.subplots(nrows = num_plots_per_page, ncols = 1)
                for cur_ax in ax:
                    cur_ax.set_xticks([])
                    cur_ax.set_yticks([])
        
            ax[current_plot_index].set_xlim(0 - 1/(linewidth/2), 1 + 1/(linewidth/2))
            plot_ranks = []

            for x in plot:
                ax[current_plot_index].axvline( ( 1/ sorted_len) * x['rank'], linewidth = (100 / sorted_len) * 3.5)
                plot_ranks.append(x['rank'])

            in_group =  [rank_to_value[x] for x in plot_ranks]
            out_group = [rank_to_value[x] for x in range(0, len(sorted_results)) if not x in plot_ranks]
            ks_stat, pvalue = stats.ks_2samp(in_group, out_group)
            ax[current_plot_index].set_title('p-value: ' + str(round(pvalue,3)) + '  ks stat: ' + str(round(ks_stat,3)), loc='left')

            current_plot_index += 1

            print('plot')
            print(plot)
            print(plot_ranks)
            print(in_group)
            print(out_group)

        plt.savefig('sav.png')

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
