MINORITY_CLASS = "MIN"
MAJORITY_CLASS = "MAJ"
EMPTY_CUBE = "The cube is empty"


class Cube(object):
    def __init__(self, coordinates):
        self.coordinates = [coordinates]
        self.data = []
        self.num_of_minority = None

    def join(self, cube):
        self.coordinates.extend(cube.coordinates)
        self.data.extend(cube.data)
        if not (self.num_of_minority is None or cube.num_of_minority is None):
            self.num_of_minority += cube.num_of_minority
        else:
            self.num_of_minority = None

    def add_datapoint(self, element):
        self.data.append(element)
        self.num_of_minority = None

    def num_of_class_examples(self, clazz):
        if self.num_of_minority is None:
            self.num_of_minority = sum([1 for element in self.data if element.clazz == clazz])
        if clazz == MINORITY_CLASS:
            return self.num_of_minority
        return self.num_of_examples()-self.num_of_minority

    def num_of_examples(self):
        return len(self.data)

    def prop_of_minority(self):
        if self.num_of_examples() > 0:
            return float(self.num_of_class_examples(MINORITY_CLASS)) / self.num_of_examples()
        return EMPTY_CUBE
