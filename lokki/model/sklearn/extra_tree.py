import numpy as np

from sklearn.tree import ExtraTreeClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from lokki.model import ModelChoice

class ExtraTree(ModelChoice):

    def __init__(self):
        pass

    def evaluate(self, parameters, X_train, X_test, y_train, y_test):

        model = ExtraTreeClassifier(max_depth = 10)
        model.fit(X_train, y_train)
        score = None
        pred = model.predict(X_test)

        if parameters['metric'] == 'auc':
            score = roc_auc_score(np.array(y_test) == pred[0], pred == pred[0])

        elif parameters['metric'] == 'precision':
            score = precision_score(np.array(y_test) == pred[0], pred == pred[0])

        elif parameters['metric'] == 'recall':
            score = recall_score(np.array(y_test) == pred[0], pred == pred[0])

        return score

    def get_name(self):
        return 'Extra_Tree'
