from clustering.cube_classifiers.cube_classifier import CubeClassifier
from clustering.cube import EMPTY_CUBE, MINORITY_CLASS
from clustering.cube_classifiers.cube_classifier import SAFE, BORDERLINE, RARE, OUTLIER, ONLY_MAJORITY


class ProbabilityCubeClassifier(CubeClassifier):
    def __init__(self,):
        pass

    @staticmethod
    def class_of_cube(cube):
        prop = cube.prop_of_minority()
        if prop == EMPTY_CUBE:
            return prop
        stats = cube.coordinates, cube.num_of_class_examples(MINORITY_CLASS), cube.num_of_examples(), prop
        if prop > 0.7:
            return SAFE, stats
        elif prop > 0.3:
            return BORDERLINE, stats
        elif prop > 0.1:
            return RARE, stats
        elif prop > 0:
            return OUTLIER, stats
        else:
            return ONLY_MAJORITY, stats
