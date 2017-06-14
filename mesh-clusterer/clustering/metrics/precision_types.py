from clustering.metrics.metric import Metric
from clustering.cube_classifiers.probability_cube_classifier import RARE, OUTLIER, BORDERLINE, SAFE
from sklearn import metrics


class ElementsTypePrecision(Metric):
    def __init__(self, mesh):
        self.mesh = mesh

    def calculate_precision_mesh_clusterer(self):
        pred, real = self.mesh.convert_to_lists()
        precision_to_return = list()
        for type_of_examples in [SAFE, BORDERLINE, RARE, OUTLIER]:
            if type_of_examples not in pred:
                precision_to_return.append('-')
            else:
                precisions = metrics.precision_score(real, pred, labels=[type_of_examples], average=None)
                precision_to_return.append(precisions[0])
        return precision_to_return  # [safe_prec, borderline_prec, rare_prec, outlier_prec]
