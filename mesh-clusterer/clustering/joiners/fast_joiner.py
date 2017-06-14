from clustering.joiners.mesh_joiner import MeshJoiner


class FastJoiner(MeshJoiner):
    def __init__(self, clazz):
        self.to_join = None
        self.clazz = clazz
        self.analyzed_cubes = set()

    def get_to_join(self, mesh):
        analyzed_cubes = set()
        to_join = [cube for cube in mesh.get_cubes() if
                   cube.num_of_class_examples(self.clazz) == cube.num_of_examples() and cube.num_of_examples() != 0]
        for cube in to_join:
            if cube in analyzed_cubes:
                continue
            for neighbour in mesh.get_neighbours(cube):
                if neighbour in to_join:
                    analyzed_cubes.add(neighbour)
                    yield (cube, neighbour)
            analyzed_cubes.add(cube)
        yield (None, None)

    def find_next_to_join(self, mesh):
        if self.to_join is None:
            self.to_join = self.get_to_join(mesh)
        return self.to_join.next()
