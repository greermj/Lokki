import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from itertools import combinations 

from lokki.analyze import ModelTransformSets
from lokki.lib import ks_test

class Enrichment:

    def __init__(self, analysis_object):
        self.analysis_object = analysis_object

        self.max_combn = 2

    def run(self, filename):
    
        results = dict() 
        ranked_data = self.get_ranked_list()
        dimensions = list(set([x for y in ranked_data for x in y['key']]))

        # Add single dimension data 
        for dimension in dimensions:
            print(dimension)
            enrichment_bars = []
            for data in ranked_data:
                if dimension in data['key']:
                    enrichment_bars.append(1)
                else:
                    enrichment_bars.append(0)
            print(enrichment_bars)
            ks_stat, p_value = ks_test(ranked_data, enrichment_bars)
            results[dimension] = {'name' : dimension, 'bars' : enrichment_bars.copy(), 'ks_stat' : ks_stat, 'p_value' : p_value}

        # Add combination dimension data
        for i in range(2, self.max_combn + 1):

            for combination in combinations(dimensions, i):
                print(combination)
                enrichment_bars = results[combination[0]]['bars']
                for j in range(len(combination)):
                    enrichment_bars = list(np.array(enrichment_bars) & np.array(results[combination[j]]['bars'])) 
                print(enrichment_bars)
                ks_stat, p_value = ks_test(ranked_data, enrichment_bars)
                results[combination] = {'name' : combination, 'bars' : enrichment_bars.copy(), 'ks_stat' : ks_stat, 'p_value' : p_value}

        # Sort then output 
        sorted_results = sorted(results.items(), key = lambda x : x[1]['p_value'])

        for i in range(10):
            key = sorted_results[i][0]
            values = sorted_results[i][1]

            plt.xlim(0, len(values['bars']))
            plt.ylim(0, 1)
            plt.bar(range(0, len(values['bars'])), values['bars'], width = 1, color = 'k')
            plt.xticks([]),plt.yticks([]) 
            plt.title('p-value: ' + str(round(values['p_value'], 4)) + '    stat: ' + str(round(values['ks_stat'], 4)), loc = 'left', fontsize = 10)
            plt.ylabel(values['name'])
            plt.savefig('sav_' + str(i) + '.png')

        return sorted(results.items(), key = lambda x : x[1]['p_value'])
                
    def get_ranked_list(self):
        return sorted(self.analysis_object.results, key = lambda x : x['value'], reverse = True)
