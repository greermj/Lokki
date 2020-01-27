from sklearn.linear_model import LogisticRegression

from lokki.model import ModelChoice

class LogisticRegression(ModelChoice):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def get_name(self):
        return 'Logistic Regression'
