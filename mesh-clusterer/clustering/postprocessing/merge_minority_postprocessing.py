from clustering.cube import MINORITY_CLASS


class MergeMinorityPostprocessing:
    def __init__(self):
        self.analyzed_cubes = set()

    def postprocess(self, mesh):
        while True:
            cube1, cube2 = self.next_to_join(mesh)
            if cube1 is None or cube2 is None:
                break
            mesh.join_cubes(cube1, cube2)
        return mesh

    def get_num_of_minority(self, cube):
        return sum(1 for elem in cube.data if elem.clazz == MINORITY_CLASS)

    def is_available_to_merge(self, cube):
        return self.get_num_of_minority(cube) != 0

    def next_to_join(self, mesh):
        for cube in mesh.get_cubes():
            if cube in self.analyzed_cubes:
                continue
            if self.is_available_to_merge(cube):
                for neighbour in mesh.get_neighbours(cube):
                    if self.is_available_to_merge(neighbour):
                        return cube, neighbour
            self.analyzed_cubes.add(cube)
        return None, None
