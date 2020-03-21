import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from lokki.analyze import ModelTransformSets

class Enrichment:

    def __init__(self, analysis_object):
        self.analysis_object = analysis_object
        #self.results = results 
        #self.enrichment_sets,  self.aggregate_sets, self.custom_sets = ModelTransformSets(results).get_model_transform_sets()


        # Plot configurations
        self.offset = 0.0015
        self.linewidth =38
        self.unit = 0.003

    def run(self, filename):

        dimensions = list(set([x.lower() for y in self.analysis_object.results for x in y['key']]))

        print(dimensions)


        for dimension in dimensions:

            # Extract all results that contain dimension 
            dimension_results = [x for x in self.analysis_object.results if dimension.lower() in [y.lower() for y in x['key']]]

            pass

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
        return sorted(results.results, key = lambda x : x['value'], reverse = True)
