# Notes: Should run data through each grid transform and then the model and return the top performance 
class ModelTransformAnalysis:
    
    def __init__(self, transform_instance, model_instance, parameters):

        self.transform_instance = transform_instance
        self.model_instance = model_instance
        self.parameters = parameters

    # For each param in grid run transform then get and store model performance (keep track of max)
    def get_performance(self, dataset):
        
