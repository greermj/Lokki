import numpy as np

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from skopt import BayesSearchCV
from skopt.space import Integer, Real

from lokki.model import ModelChoice

class AdaBoost(ModelChoice):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = self.get_model(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, parameters, X_train, X_test, y_train, y_test):

        model = self.get_model(X_train, y_train)

        score = None
        pred = model.predict(X_test)
        if parameters['metric'] == 'auc':
            pred = model.predict(X_test)
            score = roc_auc_score(np.array(y_test) == pred[0], pred == pred[0])

        elif parameters['metric'] == 'precision':
            pred = model.predict(X_test)
            score = precision_score(np.array(y_test) == pred[0], pred == pred[0])

        elif parameters['metric'] == 'recall':
            pred = model.predict(X_test)
            score = recall_score(np.array(y_test) == pred[0], pred == pred[0])

        return score

    def get_model(self, X, y):
        search_space = {'n_estimators'  : Integer(5, 1000),
                        'learning_rate' : Real(0.01, 2.0)}
        model = BayesSearchCV(AdaBoostClassifier(random_state = 0), search_space, random_state = 0, n_iter = 1, cv = 3, n_jobs = -1)
        model.fit(X, y)
        return model 

    def get_name(self):
        return 'AdaBoost'
