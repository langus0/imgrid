from clustering.mesh.mesh import Mesh
from clustering.cube import Cube, MINORITY_CLASS
from clustering.cube_classifiers.cube_classifier import ONLY_MAJORITY
from clustering.cube_classifiers.probability_cube_classifier import ProbabilityCubeClassifier

__author__ = "Sebastian Firlik"


class FakeMesh(Mesh):
    """
    Class representing Mesh for k-means and DBSCAN results
    """

    def __init__(self, data, clusters):
        self.mesh = {}
        self.create_mesh(data, clusters)

    def create_mesh(self, data, clusters):
        how_many_cubes = max(clusters) + 1
        for i in range(how_many_cubes):
            self.mesh[i] = Cube(i)
        for element, cluster_number in zip(data, clusters):
            if cluster_number != -1:
                self.mesh[cluster_number].add_datapoint(element)

    def get_cubes(self):
        return set(self.mesh.itervalues())

    def convert_to_lists(self):
        pred = list()
        real = list()
        for cube in self.get_cubes():
            typ = ProbabilityCubeClassifier.class_of_cube(cube)[0]
            if typ == ONLY_MAJORITY:
                continue
            for element in cube.data:
                if element.clazz == MINORITY_CLASS:
                    pred.append(typ)
                    real.append(element.typeOfExampe)

        return pred, real


class FakeMeshTypesPregenerated(Mesh):
    """
    Class representing Mesh for k-means and DBSCAN results
    """

    def __init__(self, data, predicted_types):
        self.mesh = {}
        self.predicted_types = predicted_types
        self.create_mesh(data)

    def create_mesh(self, data):
        for id, element in enumerate(data):
            self.mesh[id] = Cube(id)
            self.mesh[id].add_datapoint(element)

    def get_cubes(self):
        return set(self.mesh.itervalues())

    def convert_to_lists(self):
        pred = list()
        real = list()
        for id, cube in self.mesh.iteritems():
            typ = self.predicted_types[id]
            if typ == ONLY_MAJORITY:
                continue
            for element in cube.data:
                if element.clazz == MINORITY_CLASS:
                    pred.append(typ)
                    real.append(element.typeOfExampe)
        return pred, real