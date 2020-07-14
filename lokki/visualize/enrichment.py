import sys
import os

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scipy import stats
from itertools import combinations 
from scipy.stats import ks_2samp
import colorsys

from lokki.lib import PipelineComponents


# Description: Returns orthogonal colors 
def get_colors(number_of_colors):
    result=[]
    for i in np.arange(0., 360., 360. / number_of_colors):
        hue = i/360.
        lightness  = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        result.append( colorsys.hls_to_rgb(hue, lightness, saturation) )
    return result

# Description: Adds a colored dot as a ylabel 
def plot_ylabel(ax, list_of_strings, list_of_colors, **kw):
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
    bbox_anchor = (-0.07, -0.12) if len(list_of_strings) == 2 else (-0.07, 0.25)
    boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                 for text,color in zip(list_of_strings[::-1],list_of_colors) ]
    ybox = VPacker(children=boxes,align="center", pad=0, sep=0)
    anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0, frameon=False, bbox_to_anchor=bbox_anchor, 
                                      bbox_transform=ax.transAxes, borderpad=0)
    ax.add_artist(anchored_ybox)

class Enrichment:

    def __init__(self, analysis_object, kwargs):

        self.analysis_object = analysis_object

        # Set arguments using kwargs if found else set using default 
        self.max_combn = kwargs['max_combn'] if 'max_combn' in kwargs else 2
        self.min_hits  = kwargs['min_hits']  if 'min_hits'  in kwargs else 1
        self.max_combn = kwargs['max_combn'] if 'max_combn' in kwargs else 2
        self.filters   = kwargs['filters']   if 'filters'   in kwargs else None
        self.mode      = kwargs['mode']      if 'mode'      in kwargs else 'single'
        self.order     = kwargs['order']     if 'order'     in kwargs else 'asc'
        self.num       = kwargs['num']       if 'num'       in kwargs else 'all'

        # Creates output directory to store enrichment plots (note: I create separate run folders so the user wont overwrite existing figures)
        count = 0
        while True:
            if not os.path.exists('./enrichment_figs/run_' + str(count) + '/'):
                os.makedirs('./enrichment_figs/run_' + str(count))
                self.output_directory = './enrichment_figs/run_' + str(count)
                break
            else:
                count += 1

        # Get mapping between component name and component type
        self.component_name_to_type = PipelineComponents.get_name_to_component_map('')

    def run(self):
    
        results = dict() 
        ranked_data = self.get_ranked_list()
        ranked_values = [x['value'] for x in ranked_data]
        dimensions = sorted(list(set([x for y in ranked_data for x in y['key']])))

        # Create list of orthogonal colors then associate a dimension with the rgb values 
        color_options = get_colors(len(dimensions))
        color_map = dict()
        for i, x in enumerate(dimensions):
            color_map[x] = color_options[i]

        # For each dimension string (e.g. "pca")
        for dimension in dimensions:

            # Create a list of 0's and 1's indicating the presence or absence of the dimension (eg if pca appeared in the 1st and 3rd positions -> [1 0 1 0 0 0])
            enrichment_bars = []
            for data in ranked_data:
                if dimension in data['key']:
                    enrichment_bars.append(1)
                else:
                    enrichment_bars.append(0)

            # Create a list of the ranks (ie 1 -> 1st best, 4 -> 4th best, etc) for the ranks of the current dimension and then every other dimension 
            enrichment_ranks = [i for i, x in enumerate(ranked_data) if enrichment_bars[i] == 1]
            other_ranks        = [j for j in range(len(ranked_data)) if not j in enrichment_ranks]

            # If the ranks are in general less thna the other ranks then they cluster to the left and the sign of ks should be positive 
            ks_sign = np.nan
            if len(enrichment_ranks) != 0:
                ks_sign          = 1 if np.mean(enrichment_ranks) < np.median(other_ranks) else -1

            # If enrichment_ranks and other_ranks are both not empty use the scipy method else return (0, 1) which is the worst ks stat a pvalue possible 
            ks_stat, p_value = ks_2samp(enrichment_ranks, other_ranks) if enrichment_ranks and other_ranks else (0, 1) 

            # Store results as dictionary 
            results[dimension] = {'name' : dimension, 'bars' : enrichment_bars.copy(), 'ks_stat' : ks_stat * ks_sign, 'p_value' : p_value}

        # If you are analyzing more than a single factor 
        if self.mode.lower() == 'dual':

            # For each combination (ie n choose k)
            for combination in combinations(dimensions, 2):

                # Assign enrichment bars from the first dimension 
                enrichment_bars = results[combination[0]]['bars']
                for j in range(len(combination)):

                    # Update the enrichment bars by performing an & operation with every other dimension (ie the final result will be 1's when every dimension is present)
                    enrichment_bars = list(np.array(enrichment_bars) & np.array(results[combination[j]]['bars'])) 

                # See above (ie same procedure for single dimension used below)
                enrichment_ranks = [i for i, x in enumerate(ranked_data) if enrichment_bars[i] == 1]
                other_ranks      = [j for j in range(len(ranked_data)) if not j in enrichment_ranks]

                ks_sign = np.nan
                if len(enrichment_ranks) != 0:
                    ks_sign          = 1 if np.mean(enrichment_ranks) < np.median(other_ranks) else -1

                ks_stat, p_value = ks_2samp(enrichment_ranks, other_ranks) if enrichment_ranks and other_ranks else (0, 1) 

                results[combination] = {'name' : combination, 'bars' : enrichment_bars.copy(), 'ks_stat' : ks_stat * ks_sign, 'p_value' : p_value}

        # Initial sort 
        highest_scores = sorted(results.items(), key = lambda x : x[1]['ks_stat'], reverse = True)
        lowest_scores = sorted(results.items(), key = lambda x : x[1]['ks_stat'])

        # If user provides filters, only include those results that include filter elements (eg filters = ['pca'] will only include results with 'pca')
        if self.filters:
            highest_scores = [x for x in highest_scores if np.any([y in self.filters for y in x[0]]) or self.filters[0] == x[0]]
            lowest_scores  = [x for x in lowest_scores  if np.any([y in self.filters for y in x[0]]) or self.filters[0] == x[0]]

        # Only include results that have x number of hits 
        highest_scores = [x for x in highest_scores if sum(x[1]['bars']) >= self.min_hits]
        lowest_scores  = [x for x in lowest_scores  if sum(x[1]['bars']) >= self.min_hits]

        scores = lowest_scores if self.order.lower() == 'asc' else highest_scores

        # Loop through the plots to determine how many of each component exists. This is necessary to create a dynamic color mapping based on components 
        for i, plot_data in enumerate(scores):
            if isinstance(self.num, int) and i >= self.num:
                break
            key = plot_data[1]['name']
            print(key)
            if isinstance(key, tuple):
                print([self.component_name_to_type[x] for x in key])
            else:
                print(self.component_name_to_type[key])
            print()

        print('hello')

        '''
        # Create enrichment plot
        for i, plot_data in enumerate(scores):

            if isinstance(self.num, int) and i >= self.num:
                break

            key = plot_data[1]['name']
            name = '_'.join(key) if isinstance(key, tuple) else key
            num_factors = len(key) if isinstance(key, tuple) else 1
            factor_colors = (color_map[x] for x in key) if isinstance(key, tuple) else (color_map[key],)
            values = plot_data[1]
            pvalue  = round(values['p_value'], 4)

            fig, ax = plt.subplots(1, figsize=(15, 2))
            ax.bar(range(0, len(values['bars'])), values['bars'], width = 1, color = 'k')
            ax.set_xlim(0, len(values['bars']))
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])

            #plot_ylabel(ax, ('•',) * num_factors, factor_colors, size=80, weight='bold')
            #plot_ylabel(ax, ('●',) * num_factors, factor_colors, size=50, weight='bold')
            #plot_ylabel(ax, ('■',) * num_factors, factor_colors, size=50, weight='bold')
            plot_ylabel(ax, ('▶',) * num_factors, factor_colors, size=50, weight='bold')
            ax.set_title('p-value: ' + str(pvalue if pvalue > 0.01 else '< 0.01') + '    stat: ' + str(round(values['ks_stat'], 4)), loc = 'left', fontsize = 12,  fontweight='bold')
            plt.savefig(self.output_directory + '/' + name.lower() + '.png')
            plt.close()

        # Output legend
        patches = []
        plt.figure(figsize=(4,8))
        for color_name, color_values in color_map.items():
            #patches.append(mpatches.Patch(color = color_values, label = color_name))
            patches.append(mpatches.Polygon([[0,0],[0,1],[1,0]], color = color_values, label = color_name))
        plt.legend(handles=patches, loc='center')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(self.output_directory + '/legend.png')
        plt.clf()
        '''

               
    def get_ranked_list(self):
        return sorted(self.analysis_object.results, key = lambda x : x['value'], reverse = True)
