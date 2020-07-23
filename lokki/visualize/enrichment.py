import sys
import colorsys
import os

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scipy import stats
from itertools import combinations 
from scipy.stats import ks_2samp
from matplotlib.lines import Line2D

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
    bbox_anchor = (-0.07, 0.05) if len(list_of_strings) == 2 else (-0.07, 0.25)
    boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom',rotation=90,**kw)) 
                 for text,color in zip(list_of_strings, list_of_colors)]
    ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
    anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0, frameon=False, bbox_to_anchor=bbox_anchor, 
                                      bbox_transform=ax.transAxes, borderpad=0)
    ax.add_artist(anchored_ybox)

class Enrichment:

    def __init__(self, analysis_object, kwargs):

        self.analysis_object = analysis_object

        # Set arguments using kwargs if found else set using default 
        self.min_hits  = kwargs['min_hits']  if 'min_hits'  in kwargs else 1
        self.filters   = kwargs['filters']   if 'filters'   in kwargs else None
        self.mode      = kwargs['mode']      if 'mode'      in kwargs else 'single'
        self.order     = kwargs['order']     if 'order'     in kwargs else 'asc'
        self.num       = kwargs['num']       if 'num'       in kwargs else 'all'

        # Creates output directory to store enrichment plots (note: I create separate run folders so the user wont overwrite existing figures)
        count = 0
        while True:
            if not os.path.exists('./enrichment_figs/run_' + str(count) + '/'):
                os.makedirs('./enrichment_figs/run_' + str(count))
                self.output_directory = './enrichment_figs/run_' + str(count) + '/'
                break
            else:
                count += 1

        # Get mapping between component name and component type
        self.component_name_to_type = PipelineComponents.get_name_to_component_map('')

    def run(self):
    
        results = dict() 
        ranked_data = self.get_ranked_list()
        custom_data_flag = 'grid' in ranked_data[0] # Custom datasets will not have a grid parameter 
        ranked_values = [x['value'] for x in ranked_data]
        dimensions = sorted(list(set([x for y in ranked_data for x in y['key']])))

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
            other_ranks      = [j for j, x in enumerate(ranked_data) if not j in enrichment_ranks]

            enrichment_scores = [x['value'] for i, x in enumerate(ranked_data) if enrichment_bars[i] == 1]
            other_scores      = [x['value'] for j, x in enumerate(ranked_data) if j in other_ranks]

            # If the ranks are in general less thna the other ranks then they cluster to the left and the sign of ks should be positive 
            ks_sign = np.nan
            if len(enrichment_scores) != 0:
                ks_sign          = 1 if np.mean(enrichment_scores) > np.mean(other_scores) else -1

            # If enrichment_ranks and other_ranks are both not empty use the scipy method else return (0, 1) which is the worst ks stat a pvalue possible 
            ks_stat, p_value = ks_2samp(enrichment_scores, other_scores) if enrichment_ranks and other_ranks else (0, 1) 

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
                other_ranks      = [j for j, x in enumerate(ranked_data) if not j in enrichment_ranks]

                enrichment_scores = [x['value'] for i, x in enumerate(ranked_data) if enrichment_bars[i] == 1]
                other_scores      = [x['value'] for j, x in enumerate(ranked_data) if j in other_ranks]

                ks_sign = np.nan
                if len(enrichment_scores) != 0:
                    ks_sign          = 1 if np.mean(enrichment_scores) > np.mean(other_scores) else -1

                ks_stat, p_value = ks_2samp(enrichment_scores, other_scores) if enrichment_ranks and other_ranks else (0, 1) 

                results[combination] = {'name' : combination, 'bars' : enrichment_bars.copy(), 'ks_stat' : ks_stat * ks_sign, 'p_value' : p_value}

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


        if custom_data_flag:
            self.draw_de_novo_analysis_enrichment_plots(scores)
        else:
            self.draw_precomputed_analysis_enrichment_plots(scores)
            

    def draw_precomputed_analysis_enrichment_plots(self, scores):
        # Create enrichment plot
        for i, plot_data in enumerate(scores):

            if isinstance(self.num, int) and i >= self.num:
                break

            key = plot_data[1]['name']
            values = plot_data[1]
            pvalue  = round(values['p_value'], 4)
            name = '_'.join(key) if isinstance(key, tuple) else key
            ylabel = '\n'.join(key) if isinstance(key, tuple) else key

            fig, ax = plt.subplots(1, figsize=(15, 2))
            ax.bar(range(0, len(values['bars'])), values['bars'], width = 1, color = 'k')
            ax.set_xlim(0, len(values['bars']))
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(ylabel)
            ax.set_title('p-value: ' + str(pvalue if pvalue > 0.01 else '< 0.01') + '    stat: ' + str(round(values['ks_stat'], 4)), loc = 'left', fontsize = 12,  fontweight='bold')
            plt.savefig(self.output_directory + '/' + name.lower() + '.png')
            plt.close()


    def draw_de_novo_analysis_enrichment_plots(self, scores):

        # Loop through the plots to determine how many of each component exists. This is necessary to create a dynamic color mapping based on components
        component_sets = {'data_transform' : set(), 'feature_transform' : set(), 'model' : set()}
        for i, plot_data in enumerate(scores):
            if isinstance(self.num, int) and i >= self.num:
                break
            key = plot_data[1]['name']
            if isinstance(key, tuple):
                component_types = {x : self.component_name_to_type[x] for x in key}
                for x, y in component_types.items():
                    component_sets[y].add(x)
            else:
                component_type = self.component_name_to_type[key]
                component_sets[component_type].add(key)
    
        # Generate the color map between the component type options and a set of orthogonal colors 
        color_map = self.get_loaded_color_map(component_sets)

        # Create enrichment plot
        for i, plot_data in enumerate(scores):

            if isinstance(self.num, int) and i >= self.num:
                break

            key = plot_data[1]['name']
            values = plot_data[1]
            pvalue  = round(values['p_value'], 4)
            name = '_'.join(key) if isinstance(key, tuple) else key
            factor_labels, factor_colors = self.get_label(key, color_map)

            fig, ax = plt.subplots(1, figsize=(15, 2))
            ax.bar(range(0, len(values['bars'])), values['bars'], width = 1, color = 'k')
            ax.set_xlim(0, len(values['bars']))
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('p-value: ' + str(pvalue if pvalue > 0.01 else '< 0.01') + '    stat: ' + str(round(values['ks_stat'], 4)), loc = 'left', fontsize = 12,  fontweight='bold')

            plot_ylabel(ax, factor_labels, factor_colors, size=50, weight='bold')
            plt.savefig(self.output_directory + '/' + name.lower() + '.png')
            plt.close()

        # Output legend
        fig, ax = plt.subplots(nrows = len(color_map), ncols = 1, figsize = (4, 12))

        for i, (component_type, component_name_color_map) in enumerate(color_map.items()):

            patches = []

            if component_type == 'data_transform':
                legend_shape = 'o'
            elif component_type == 'feature_transform':
                legend_shape = 's'
            elif component_type == 'model':
                legend_shape = '^'
            else:
                sys.exit("ERROR: Couldn't find component type check enrichment.py")
        
            for color_name, color_values in component_name_color_map.items():
                patches.append(Line2D([0], [0], marker = legend_shape, color = 'w', label = color_name, markerfacecolor = color_values, markersize = 15))

            if len(color_map) > 1:
                ax[i].legend(handles = patches, loc='center')
                ax[i].set_xticks([])
                ax[i].set_yticks([])
            else:
                ax.legend(handles = patches, loc='center')
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.savefig(self.output_directory + '/legend.png')
        plt.close()

    # Description: Returns shape and color for y label 
    def get_label(self, key, color_map):
        factor_labels, factor_colors = [], []

        if isinstance(key, tuple):
            for x in key:
                component_type = self.component_name_to_type[x]
                if component_type == 'data_transform':
                    factor_labels.append('●')
                elif component_type == 'feature_transform':
                    factor_labels.append('■')
                elif component_type == 'model':
                    factor_labels.append('▲')
                else:
                    sys.exit("ERROR: Couldn't find label shape check enrichment.py (tuple)")
                factor_colors.append(color_map[component_type][x])
        else:
            component_type = self.component_name_to_type[key]
            if component_type == 'data_transform':
                factor_labels.append('●')
            elif component_type == 'feature_transform':
                factor_labels.append('■')
            elif component_type == 'model':
                factor_labels.append('▲')
            else:
                sys.exit("ERROR: Couldn't find label shape check enrichment.py (string)")
            factor_colors.append(color_map[component_type][key])

        return tuple(factor_labels), tuple(factor_colors)

    # Description: Return the following mapping: {eg data_transform : { zscore : color, log : color, ...}, ...}
    def get_loaded_color_map(self, component_sets):

        color_map = dict()

        for component_type, component_set in component_sets.items():

            if len(component_set) == 0:
                continue 

            if not component_type in color_map:
                color_map[component_type] = dict()

            color_options = get_colors(len(component_set)) 

            for i, x in enumerate(component_set):
                color_map[component_type][x] = color_options[i]
        
        return color_map

    # Description: Returns ranked list by key        
    def get_ranked_list(self):
        return sorted(self.analysis_object.results, key = lambda x : x['value'], reverse = True)
