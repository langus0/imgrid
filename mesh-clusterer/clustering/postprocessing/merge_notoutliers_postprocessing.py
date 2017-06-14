from clustering.postprocessing.merge_minority_postprocessing import MergeMinorityPostprocessing
from clustering.cube_classifiers.cube_classifier import ONLY_MAJORITY, OUTLIER
from clustering.cube_classifiers.probability_cube_classifier import ProbabilityCubeClassifier


class MergeNotOutliersPostprocessing(MergeMinorityPostprocessing):

    def is_available_to_merge(self, cube):
        return ProbabilityCubeClassifier.class_of_cube(cube)[0] not in [ONLY_MAJORITY, OUTLIER]
