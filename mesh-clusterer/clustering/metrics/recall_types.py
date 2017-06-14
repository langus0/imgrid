from clustering.metrics.metric import Metric
from clustering.cube_classifiers.probability_cube_classifier import RARE, OUTLIER, BORDERLINE, SAFE
from sklearn import metrics


class ElementsTypeRecall(Metric):
    def __init__(self, mesh):
        self.mesh = mesh

    def calculate_recall_mesh_clusterer(self):
        pred, real = self.mesh.convert_to_lists()

        recall_to_return = list()
        for index, type_of_examples in enumerate([SAFE, BORDERLINE, RARE, OUTLIER]):
            if type_of_examples not in real:
                recall_to_return.append('-')
            else:
                recalls = metrics.recall_score(real, pred, labels=[type_of_examples], average=None)
                recall_to_return.append(recalls[0])

        return recall_to_return  # [safe_recall, borderline_recall, rare_recall, outlier_recall]
