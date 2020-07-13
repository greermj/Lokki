import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from lokki.model import ModelChoice

class DecisionTree(ModelChoice):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = DecisionTreeClassifier(max_depth = 10)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, parameters, X_train, X_test, y_train, y_test):

        model = DecisionTreeClassifier(max_depth = 10)
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
        return 'Decision_Tree'
