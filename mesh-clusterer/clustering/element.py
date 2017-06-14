from clustering.cube import MINORITY_CLASS, MAJORITY_CLASS


class Element(object):
    def __init__(self, data, clazz, typeOfExampe, clusterIdx):
        self.data = data
        self.clazz = clazz
        self.typeOfExampe = typeOfExampe
        self.clusterIdx = clusterIdx
        assert self.clazz in [MINORITY_CLASS, MAJORITY_CLASS]
