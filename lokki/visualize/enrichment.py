import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from itertools import combinations 
from scipy.stats import ks_2samp

class Enrichment:

    def __init__(self, analysis_object):
        self.analysis_object = analysis_object

        self.max_combn = 2

    def run(self, filename):
    
        results = dict() 
        ranked_data = self.get_ranked_list()
        ranked_values = [x['value'] for x in ranked_data]
        dimensions = list(set([x for y in ranked_data for x in y['key']]))

        # Add single dimension data 
        for dimension in dimensions:
            #print(dimension)
            enrichment_bars = []
            for data in ranked_data:
                if dimension in data['key']:
                    enrichment_bars.append(1)
                else:
                    enrichment_bars.append(0)
            #print(enrichment_bars)

            enrichment_ranks = [i for i, x in enumerate(ranked_data) if enrichment_bars[i] == 1]
            all_ranks        = [j for j in range(len(ranked_data)) if not j in enrichment_ranks]
            ks_sign          = 1 if np.mean(enrichment_ranks) < np.median(all_ranks) else -1
            
            ks_stat, p_value = ks_2samp(enrichment_ranks, all_ranks) if not enrichment_ranks == [] else (0, 1) 
            results[dimension] = {'name' : dimension, 'bars' : enrichment_bars.copy(), 'ks_stat' : ks_stat * ks_sign, 'p_value' : p_value}

        # Add combination dimension data
        for i in range(2, self.max_combn + 1):

            for combination in combinations(dimensions, i):
                #print(combination)
                enrichment_bars = results[combination[0]]['bars']
                for j in range(len(combination)):
                    enrichment_bars = list(np.array(enrichment_bars) & np.array(results[combination[j]]['bars'])) 
                #print(enrichment_bars)

                enrichment_ranks = [i for i, x in enumerate(ranked_data) if enrichment_bars[i] == 1]
                all_ranks        = [j for j in range(len(ranked_data)) if not j in enrichment_ranks]
                ks_sign          = 1 if np.mean(enrichment_ranks) < np.median(all_ranks) else -1

                ks_stat, p_value = ks_2samp(enrichment_ranks, all_ranks) if not enrichment_ranks == [] else (0, 1) 
                results[combination] = {'name' : combination, 'bars' : enrichment_bars.copy(), 'ks_stat' : ks_stat * ks_sign, 'p_value' : p_value}

        # Sort then output (First by p-value, then by decreasing number of bars (ie reciprocal of length))
        sorted_results  = [sorted(results.items(), key = lambda x : x[1]['ks_stat'], reverse = True), sorted(results.items(), key = lambda x : x[1]['ks_stat'])]

        fig, ax = plt.subplots(nrows = 10, ncols = 2,figsize=(20,20))

        print(ax.shape)

        for i in range(10):
            for j in range(2):

                key     = sorted_results[j][i][0]
                values  = sorted_results[j][i][1]

                ax[i][j].set_xlim(0, len(values['bars']))
                ax[i][j].set_ylim(0, 1)
                ax[i][j].bar(range(0, len(values['bars'])), values['bars'], width = 1, color = 'k')
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([]) 
                ax[i][j].set_title('p-value: ' + str(round(values['p_value'], 4)) + '    stat: ' + str(round(values['ks_stat'], 4)), loc = 'left', fontsize = 10)
                ax[i][j].set_ylabel(values['name'] if isinstance(values['name'], str) else '\n'.join(values['name']))

        fig.tight_layout(pad = 2)
        plt.savefig('hal.png')

        return sorted(results.items(), key = lambda x : x[1]['p_value'])
                
    def get_ranked_list(self):
        return sorted(self.analysis_object.results, key = lambda x : x['value'], reverse = True)
