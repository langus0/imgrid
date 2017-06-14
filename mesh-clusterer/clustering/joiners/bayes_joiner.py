from scipy.special import beta

from clustering.cube import MINORITY_CLASS
from clustering.joiners.chi2_joiner import Chi2Joiner


class BayesJoinerWithMinorityRule(Chi2Joiner):
    def calculate_test(self, cube1, cube2):
        if cube1.num_of_examples() == 0 or cube2.num_of_examples() == 0:
            return 0
        am = 0.5
        bm = 0.5
        af = 0.5
        bf = 0.5

        rm = cube1.num_of_class_examples(MINORITY_CLASS)
        nm = cube1.num_of_examples()

        rf = cube2.num_of_class_examples(MINORITY_CLASS)
        nf = cube2.num_of_examples()

        pierwszy = beta(am + af + rm + rf, bm + bf + nm + nf - rm - rf) / beta(am + af, bm + bf)
        drugi = beta(am + rm, bm + nm - rm) / beta(am, bm)
        trzeci = beta(af + rf, bf + nf - rf) / beta(af, bf)

        bf = pierwszy / (drugi * trzeci)

        return bf / (bf + 1)

    def calculate_chi2(self, cube1, cube2):
        if (cube1.num_of_class_examples(MINORITY_CLASS) > 0) ^ (cube2.num_of_class_examples(MINORITY_CLASS) > 0):
            if cube1.num_of_examples() == 0 or cube2.num_of_examples() == 0:  # exception for empty cubes
                return 1.
            else:
                return 0.
        return self.calculate_test(cube1, cube2)
