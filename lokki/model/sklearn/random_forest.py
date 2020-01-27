from sklearn.ensemble import RandomForestClassifier

from lokki.model import ModelChoice

class RandomForest(ModelChoice):
    
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def get_model_name(self):
        return 'Random Forest'
