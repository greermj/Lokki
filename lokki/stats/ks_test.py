from lokki.analyze.model_transform_sets import ModelTransformSets

class KSTest:

    def __init__(self, results):
        self.results = results 
        self.enrichment_sets, self.aggregate_sets, self.custom_sets = ModelTransformSets().get_model_transform_sets()

        print(self.enrichment_sets)

    def run(self, filename):
        pass
