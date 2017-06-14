from clustering.metrics.metric import Metric
from clustering.cube_classifiers.probability_cube_classifier import RARE, OUTLIER, BORDERLINE, SAFE
from sklearn import metrics


class ElementsTypeFScore(Metric):
    def __init__(self, mesh):
        self.mesh = mesh

    def calculate_f_score_mesh_clusterer(self):
        pred, real = self.mesh.convert_to_lists()
        labels = list()
        f_score = 0
        for type_of_example in [SAFE, BORDERLINE, RARE, OUTLIER]:
            if (type_of_example in pred) and (type_of_example in real):
                labels.append(type_of_example)
        if len(labels) <= 1:
            return "-"
        elif len(labels) == 2:
            f_score += metrics.f1_score(real, pred, labels=labels, pos_label=None, average='macro')
            return f_score
        else:
            f_score = metrics.f1_score(real, pred, labels=labels, pos_label=None, average='macro')  # unweighted mean
            return f_score  # just one value
