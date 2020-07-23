# Lokki: An automatic machine learning python framework for robust pipeline identification and evaluation 

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


Lokki is an open source Python automatic machine learning framework for 16s metagenomic data. The library focuses on being highly modular and extensible to allow additional models and dimensionality reduction approaches to be added seamlessly.

Under the hood, the library uses many of the APIs from existing machine learning libraries to maintain high quality data pipelines and learning models. One of the main goals of Lokki is to enable fast experimentation, by leveraging the existing tools and pipelines provided by `numpy`, `pandas`, `matplotlib`, and `sklearn`.

_The goal of this library is to enable research-quality analysis of multiple machine learning models and dimensionality reduction techniques._

Read the documentation and or walk through the quickstart tutorial below

## Guiding principles

_Inspired by [Keras' guiding principles](https://github.com/keras-team/keras)._

- **User friendliness.** Lokki is a library designed for human beings, not machines. It puts user experience front and center. Lokki follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases.

- **Modularity.** The library is composed of a small number of fully configurable modules that can be plugged together with as few restrictions as possible. In particular, model, transform, visualize, and analyze are all standalone modules that you can modify.

- **Easy extensibility.** New models and feature transformation strategies are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Lokki suitable for large-scale scientific research studies.

## Getting Started

You can get started testing on your local machine and by viewing our many examples

## Installation

Lokki requires Python >= 3.6 for all functionality to work as expected.

```bash
pip install lokki
```

## Quick Start

The following code example illustrates basic library usage

```bash

import lokki

import numpy as np
import pandas as pd
import dill as pickle

# Read data
path_to_dataset  = './docs/data/sample_data_baxter_tumor.csv'
path_to_taxonomy = './docs/data/baxter.taxonomy'

data = pd.read_csv(path_to_dataset)
taxonomy = pd.read_csv(path_to_taxonomy, sep='\t')

# Zero filter
data = data.loc[:, ((data == 0).mean() < 0.7) | np.array([x.lower().startswith('target') for x in data.columns.values])].copy()

# Configure and run 
analysis = lokki.configure(dataset = data,
                           target_name = 'target',
                           data_transforms = ['zscore'],
                           feature_transforms = ['chi_square', 'mutual_information', 'factor_analysis', 'mutual_information'],
                           models = ['decision_tree', 'random_forest', 'ridge_regression'],
                           metric = 'auc',
                           taxonomy = taxonomy)

results = analysis.run()

# Save results 
pickle.dump(results, open('results.p', 'wb'))

# Enrichment analysis visualization
lokki.plot(analysis_object = results,
           plot_type = 'enrichment',
           mode = 'dual',
           min_hits = 2,
           num = 20,
           order = 'asc')

```

**Working on your first Pull Request?** You can learn how from this _free_ series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github)

