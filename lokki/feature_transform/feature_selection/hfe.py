import re
import sys

import numpy as np
import pandas as pd

from treelib import Node, Tree
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif

from lokki.feature_transform import FeatureTransformChoice

class HFE(FeatureTransformChoice):

    def __init__(self, dataset_shape, taxonomy = None):
        self.dataset_shape = dataset_shape
        self.taxonomy      = taxonomy

    def fit(self, hyperparameters, X, y):

        # Initialize new tree
        self.tree = Tree()

        if isinstance(self.taxonomy, type(None)):
            print('Warning: Could not execute HFE algorithm no taxonomy provided')
        else:

            # Add otus to the internal tree structure 
            for i, otu in enumerate([a for a in X.columns.values if a.lower().startswith('otu')]):
                self._add_otu_to_tree(self.tree, otu, X.copy())

            # Perform filtering then determine which nodes in the tree will act as the final features 
            self._correlation_filter()
            self._path_ig_filter(y)
            self._leaf_ig_filter(y)
            self.valid_ids = [x.identifier for x in self.tree.all_nodes() if x.data['valid']]
            
    def transform(self, X, y = None):

        # Build a new tree (Note: The self.tree is built on training data, so we need a new tree for unseen data)
        new_tree = Tree()
        for i, otu in enumerate([a for a in X.columns.values if a.lower().startswith('otu')]):
            self._add_otu_to_tree(new_tree, otu, X.copy())
        
        # Extract the final features and store in dataframe
        result = pd.DataFrame()

        for i, current_id in enumerate(self.valid_ids):
            result[i] = new_tree[current_id].data['feature_vector']

        return result

    def get_name(self):
        return 'HFE'

    def hyperparameter_grid(self):
        return None

    # Description: Populates tree structure
    def _add_otu_to_tree(self, tree, otu_name, otu_table):

        raw_string       = self.taxonomy['Taxonomy'][otu_name.lower() == np.array([x.lower() for x in self.taxonomy['OTU']])].values[0]
        taxonomic_string = re.sub(r'[()0-9]', '', raw_string)
        taxonomic_levels = taxonomic_string.split(';')

        feature_vector   = otu_table.loc[:,otu_name].values.copy()
        
        for i, level in enumerate(taxonomic_levels):

            # Some of the levels might be empty if there an extra ; in the taxonomy file or two back to back. These should be skipped
            if level == '':
                continue

            if tree.contains(level):
                # Increment node OTU vector with argument OTU vector (ie node vector + new otu vector)
                tree[level].data['feature_vector'] += feature_vector.copy()

            else:

                # Create new node with dictionary of OTU vector and boolean flag indicating valid node
                tree.create_node(tag = level,
                                 identifier = level,
                                 parent = taxonomic_levels[i - 1] if i != 0 else None,
                                 data = {'feature_vector' : feature_vector.copy(), 'valid' : True})

    # Description: Removes all child nodes whose feature vector is correlated with their parent node by switching the valid flag for that node to False
    def _correlation_filter(self, threshold = 0.80):

        paths = self.tree.paths_to_leaves()

        for path in paths:

            if len(path) < 1:
                continue

            for i in range(1, len(path)):
                parent_feature_vector = self.tree[path[-1 - i + 1]].data['feature_vector']
                child_feature_vector  = self.tree[path[-1 - i]].data['feature_vector']
                current_correlation = pearsonr(parent_feature_vector, child_feature_vector)[0]

                if current_correlation > threshold:
                    self.tree[path[-1 - i]].data['valid'] = False

    # Description: Filters nodes based on average path information gain (IG)
    def _path_ig_filter(self, labels):

        paths = self.tree.paths_to_leaves()

        for path in paths:

            added = 0
            avg_path_IG = None

            for i, bacteria in enumerate(path):

                # For thoses nodes that passed the correlation filter compute the running IG average
                if self.tree[bacteria].data['valid']:
                    if added == 0:
                        avg_path_IG = mutual_info_classif(X = self.tree[bacteria].data['feature_vector'].reshape(len(labels), 1), y = labels, random_state = 0)
                        added += 1
                    else:
                        avg_path_IG = avg_path_IG * (added / (added + 1)) + (mutual_info_classif(X = self.tree[bacteria].data['feature_vector'].reshape(len(labels), 1), y = labels, random_state = 0) / (added + 1))
                        added += 1

            # If a node in the path is less than the average or it is uninformative (ie all zeros) remove the node
            for bacteria in path:

                current_bacteria_IG = mutual_info_classif(X = self.tree[bacteria].data['feature_vector'].reshape(len(labels), 1), y = labels, random_state = 0)

                if (avg_path_IG == None) or (current_bacteria_IG < avg_path_IG) or sum(self.tree[bacteria].data['feature_vector']) == 0:
                    self.tree[bacteria].data['valid'] = False

    # Description: Filter leaf nodes in incomplete paths based on global information gain (IG)
    def _leaf_ig_filter(self, labels):

        # Returns whether a path is incomplete 
        def incomplete_path(path):
            incomplete = False
            for bacteria in path:
                if not self.tree[bacteria].data['valid']:
                    incomplete = True
            return incomplete

        paths = self.tree.paths_to_leaves()
        avg_tree_IG = None
        added = 0

        # Compute global avg IG
        for path in paths:
            for bacteria in path:
                    # Only the remaining nodes contribute to global avg IG
                    if self.tree[bacteria].data['valid']:
                        if added == 0:
                            avg_tree_IG = mutual_info_classif(X = self.tree[bacteria].data['feature_vector'].reshape(len(labels), 1), y = labels, random_state = 0)
                            added += 1
                        else:
                            avg_tree_IG = avg_tree_IG * (added / (added + 1)) + (mutual_info_classif(X = self.tree[bacteria].data['feature_vector'].reshape(len(labels), 1), y = labels, random_state = 0) / (added + 1))
                            added += 1

        # The leaf nodes are the final element in the path. If the leaf node IG is zero or less than global IG and is a part of an incomplete path it should be removed 
        for path in paths:

            leaf_node_IG = mutual_info_classif(X = self.tree[path[-1]].data['feature_vector'].reshape(len(labels), 1), y = labels, random_state = 0)

            if incomplete_path(path) and ((leaf_node_IG == 0) or (leaf_node_IG < avg_tree_IG)):
                self.tree[path[-1]].data['valid'] = False
