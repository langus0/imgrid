# coding: utf-8

from clustering.mesh.mesh import Mesh
from clustering.cube import Cube
from clustering.cube_classifiers.probability_cube_classifier import ProbabilityCubeClassifier, ONLY_MAJORITY, \
    MINORITY_CLASS
import numpy as np
import math
import itertools

__author__ = "Mateusz Lango, Dariusz Brzezinski"


class EqualWidthMesh(Mesh):
    """

    """

    def __init__(self, data, k_bins, dim_mod="divide"):
        num_attributes = len(data[0].data)
        predefined_bins = False

        if k_bins == "sqrt":
            self.k = int(math.sqrt(len(data)))
        elif k_bins == "1+3logn":
            self.k = int(1 + 3.322 * math.log(len(data)))
        elif k_bins == "5logn":
            self.k = int(5 * math.log(len(data)))
        elif k_bins == "IQR":
            k_sum = 0

            for attribute_idx in xrange(num_attributes):
                attribute_data = []
                for element in data:
                    attribute_data.append(element.data[attribute_idx])

                iqr = np.subtract(*np.percentile(attribute_data, [75, 25]))
                rng = np.subtract(*np.percentile(attribute_data, [100, 0]))
                k_sum += rng / (2.64 * iqr * len(data) ** (-1. / 3.))

            self.k = int(k_sum / (num_attributes * 1.0))
        elif k_bins == "div10":
            self.k = int(len(data) / 10)
        else:
            self.k = k_bins
            predefined_bins = True

        if dim_mod is not None and not predefined_bins:
            if dim_mod == "root":
                self.k = int(math.ceil(self.k ** (1. / num_attributes)))
            elif dim_mod == "divide":
                self.k = int(math.ceil(self.k / num_attributes))
        self.create_mesh(data)

    def get_num_dimensions(self):
        return len(self.attr2intervals)

    def get_cubes(self):
        return set(self.mesh.itervalues())

    def join_cubes(self, cube1, cube2):
        cube1.join(cube2)
        for coords in cube2.coordinates:
            self.mesh[coords] = cube1

    def get_cube(self, coordinates):
        assert len(coordinates) == self.get_num_dimensions()
        return self.mesh.get(tuple(coordinates), None)

    def get_neighbours(self, cube):
        for coord in cube.coordinates:
            for i in xrange(len(coord)):
                for modifier in [-1, +1]:
                    new_coord = list(coord)
                    new_coord[i] += modifier
                    new_coord = tuple(new_coord)
                    if new_coord in cube.coordinates:
                        continue
                    neighbour = self.get_cube(new_coord)
                    if neighbour is not None:
                        assert neighbour != cube
                        yield neighbour

    def create_mesh(self, data):
        self._attr2intervals(data)
        self.mesh = {}
        for coord in itertools.product(*([range(self.k)] * self.get_num_dimensions())):
            coord = tuple(coord)
            self.mesh[coord] = Cube(coord)
        for element in data:
            self.mesh[self.get_coordinates(element)].add_datapoint(element)

    def _attr2intervals(self, data):
        self.attr2intervals = []
        for attribute_idx in xrange(len(data[0].data)):
            min = np.min([i.data[attribute_idx] for i in data])
            max = np.max([i.data[attribute_idx] for i in data])
            self.attr2intervals.append(self._create_intervals(min, max))

    def _create_intervals(self, min, max):
        return np.linspace(min, max, num=self.k + 1, endpoint=True)

    def get_coordinates(self, element):
        coordinates = []
        for attribute_idx, range in enumerate(self.attr2intervals):
            value = element.data[attribute_idx]
            for idx, start_of_interval in enumerate(range):
                if value < start_of_interval:
                    assert idx != 0
                    coordinates.append(idx - 1)
                    break
            if len(coordinates) == attribute_idx:
                coordinates.append(self.k - 1)
        return tuple(coordinates)

    def convert_to_true_coordinates(self, coordinates):
        result = []
        for idx, coord in enumerate(coordinates):
            result.append(self.attr2intervals[idx][coord])
        return result

    def get_coord_width(self, coordinates):
        result = []
        for idx in xrange(len(coordinates)):
            result.append(self.attr2intervals[idx][1] - self.attr2intervals[idx][0])
        return result

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

    def count_cubes(self):
        return len(self.get_cubes())

    def count_minority_cubes(self):
        count = 0
        for cube in self.get_cubes():
            if cube.num_of_class_examples(MINORITY_CLASS) > 0:
                count += 1
        return count

    def generate_labels(self, minority_only=False):
        return self.generate_labels_for_evaluation(minority_only)[0]

    def generate_labels_for_evaluation(self, minority_only=False):
        labels = list()
        true_labels = list()
        for index, cube in enumerate(self.get_cubes()):
            for element in cube.data:
                if minority_only and element.clazz != MINORITY_CLASS:
                    continue
                labels.append(index)
                true_labels.append(element.clusterIdx)
        return labels, true_labels

    def count_examples_of_type(self, type_of_examples):
        how_many = 0
        for cube in self.get_cubes():
            if ProbabilityCubeClassifier.class_of_cube(cube)[0] == type_of_examples:
                for element in cube.data:
                    if element.clazz == MINORITY_CLASS:
                        how_many += 1
        return how_many
