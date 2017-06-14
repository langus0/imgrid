from clustering.metrics.metric import Metric
from clustering.cube_classifiers.probability_cube_classifier import RARE, OUTLIER, BORDERLINE, SAFE
from sklearn import metrics
from numpy import roots


class GMeanMetric(Metric):
    def __init__(self, pred, real):
        self.predicted_types = pred
        self.true_types = real

    def g_mean_score(self):
        under_root = 1
        degree_of_root = 0

        for type_of_examples in [SAFE, BORDERLINE, RARE, OUTLIER]:
            true_list_to_recall = list()
            pred_list_to_recall = list()
            if type_of_examples in self.true_types:
                for i in range(len(self.predicted_types)):
                    if self.predicted_types[i] == type_of_examples:
                        pred_list_to_recall.append(1)
                    else:
                        pred_list_to_recall.append(0)
                    if self.true_types[i] == type_of_examples:
                        true_list_to_recall.append(1)
                    else:
                        true_list_to_recall.append(0)
                degree_of_root += 1
                under_root *= metrics.recall_score(true_list_to_recall, pred_list_to_recall)
        if degree_of_root == 0:
            return "-"
        g_mean = under_root**(1. / degree_of_root)
        return g_mean

