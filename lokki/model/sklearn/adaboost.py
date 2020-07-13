import numpy as np

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.model_selection import RandomizedSearchCV

from lokki.model import ModelChoice

class AdaBoost(ModelChoice):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = AdaBoostClassifier()
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, parameters, X_train, X_test, y_train, y_test):

        #ada = AdaBoostClassifier(random_state = 0)
        #distribution = {'n_estimators' : [10, 50, 100, 500, 1000, 5000]}

        #model = RandomizedSearchCV(ada, distribution, random_state = 0, n_iter = 1)
        #print(model)

        model = AdaBoostClassifier()
        model.fit(X_train, y_train)

        score = None

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

    def get_name(self):
        return 'AdaBoost'
