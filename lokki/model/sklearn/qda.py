import warnings
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from skopt import BayesSearchCV
from skopt.space import Real

from lokki.model import ModelChoice

class QDA(ModelChoice):

    def __init__(self):
        warnings.filterwarnings("ignore")

    def fit(self, X, y):
        self.model = self.get_model(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, parameters, X_train, X_test, y_train, y_test):

        model = self.get_model(X_train, y_train)

        score = None
        pred = model.predict(X_test)

        if parameters['metric'] == 'auc':
            score = roc_auc_score(np.array(y_test).astype(bool), pred.astype(bool))

        elif parameters['metric'] == 'precision':
            score = precision_score(np.array(y_test).astype(bool), pred.astype(bool))

        elif parameters['metric'] == 'recall':
            score = recall_score(np.array(y_test).astype(bool), pred.astype(bool))

        return score

    def get_model(self, X, y):
        search_space = {'reg_param' : Real(0, 1)}
        model = BayesSearchCV(QuadraticDiscriminantAnalysis(), search_space, random_state = 0, n_iter = 1, cv = 3, n_jobs = -1)
        model.fit(X, y)
        return model

    def get_name(self):
        return 'QDA'
