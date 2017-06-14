from scipy.stats import chi2_contingency, fisher_exact

from clustering.cube import MAJORITY_CLASS, MINORITY_CLASS
from clustering.joiners.mesh_joiner import MeshJoiner


class Option:
    SIMPLE = 0
    ALL_AT_ONCE = 1
    STEEPEST = 2


class Chi2Joiner(MeshJoiner):
    def __init__(self, alpha, option=Option.SIMPLE):
        self.alpha = alpha
        self.analyzed_cubes = set()
        self.option = option
        self.whole_list_to_join = None

    def find_next_to_join(self, mesh):
        if self.option == Option.SIMPLE:
            return self.find_next_to_join_simple(mesh)
        if self.option == Option.ALL_AT_ONCE:
            return self.find_next_to_join_all_at_once(mesh)
        if self.option == Option.STEEPEST:
            return self.find_next_to_join_steepest(mesh)
        raise Exception("Unknown option")

    def find_next_to_join_simple(self, mesh):
        for cube in self.order_cubes_by_minority_clazz(mesh.get_cubes()):
            if cube in self.analyzed_cubes:
                continue
            for neighbour in mesh.get_neighbours(cube):
                chi2_pvalue = self.calculate_chi2(neighbour, cube)
                if chi2_pvalue > self.alpha:
                    return cube, neighbour
            self.analyzed_cubes.add(cube)
        return None, None

    def find_next_to_join_all_at_once(self, mesh):
        if self.whole_list_to_join is None:
            self.whole_list_to_join = set([])
            for cube in mesh.get_cubes():
                for neighbour in mesh.get_neighbours(cube):
                    chi2_pvalue = self.calculate_chi2(neighbour, cube)
                    if chi2_pvalue > self.alpha:
                        self.whole_list_to_join.add((cube, neighbour))
                self.analyzed_cubes.add(cube)
            self.whole_list_to_join = list(self.whole_list_to_join)
        if len(self.whole_list_to_join) == 0:
            return None, None
        return self.whole_list_to_join.pop()

    def find_next_to_join_steepest(self, mesh):
        for cube in self.order_cubes_by_minority_clazz(mesh.get_cubes()):
            if cube in self.analyzed_cubes:
                continue

            best_neighbour = None
            best_chi2_pvalue = 0
            for neighbour in mesh.get_neighbours(cube):
                chi2_pvalue = self.calculate_chi2(neighbour, cube)

                if chi2_pvalue > best_chi2_pvalue:
                    best_chi2_pvalue = chi2_pvalue
                    best_neighbour = neighbour

            if best_chi2_pvalue > self.alpha:
                return cube, best_neighbour
            self.analyzed_cubes.add(cube)
        return None, None

    def calculate_chi2(self, cube1, cube2):
        observed = []
        for cube in [cube1, cube2]:
            cube_stats = []
            for clazz in [MAJORITY_CLASS, MINORITY_CLASS]:
                cube_stats.append(cube.num_of_class_examples(clazz))
            observed.append(cube_stats)
        if any([item == 0 for sublist in observed for item in sublist]):
            return fisher_exact(observed)[1]
        else:
            return chi2_contingency(observed)[1]

    def order_cubes_by_minority_clazz(self, cubes):
        return sorted(cubes, key=lambda cube: (cube.num_of_class_examples(MINORITY_CLASS), cube.num_of_examples()),
                      reverse=True)


class Chi2JoinerWithMinorityRule(Chi2Joiner):
    def calculate_chi2(self, cube1, cube2):
        if (cube1.num_of_class_examples(MINORITY_CLASS) > 0) ^ (cube2.num_of_class_examples(MINORITY_CLASS) > 0):
            if cube1.num_of_examples() == 0 or cube2.num_of_examples() == 0:  # exception for empty cubes
                return 1.
            else:
                return 0.
        return super(Chi2JoinerWithMinorityRule, self).calculate_chi2(cube1, cube2)
