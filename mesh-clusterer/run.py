# coding: utf-8

import os
import sys

import matplotlib

import clustering.clusterers as clusterers
import utils.clustering_util as util
import utils.plot as plot
from clustering.joiners.bayes_joiner import BayesJoinerWithMinorityRule
from clustering.joiners.chi2_joiner import Option
from clustering.mesh.equalwidth_mesh import EqualWidthMesh
from clustering.postprocessing.merge_minority_postprocessing import MergeMinorityPostprocessing

matplotlib.use('Agg')

__author__ = "Mateusz Lango, Dariusz Brzezinski"

DATASETS_PATH = os.path.join(os.path.dirname(__file__), "../datasets")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results")

CLUSTERERS = {
    clusterers.mesh_clusterer:
        [{"mesher": [EqualWidthMesh],
          "joiner": [BayesJoinerWithMinorityRule],
          "alpha": [0.75, 0.8, 0.85, 0.9, 0.95],
          "option": [Option.STEEPEST],
          "dim_mod": ["root"],
          "k_bins": ["div10"],
          "postprocessing": [MergeMinorityPostprocessing],
          "draw_plots": [False]
          }],
    clusterers.k_means_clusterer:
        [{"k_clusters": range(1, 10),
          "draw_plots": [False]}],
    clusterers.DBSCAN_clusterer:
        [{"eps": range(10, 95, 20),
          "min_samples": range(2, 3),
          "draw_plots": [False]}],
    clusterers.napierala_clusterer:
        [{"k": [5, 7, 9, 11]}]
}


def main(args):
    """
    Main method. Performs clustering and saves cluster summaries and visualizations to files.
    :param args: script arguments. Not used yet.
    """
    util.cleanup_and_prepare_folders(RESULTS_PATH)
    for root, dir, files in os.walk(os.path.abspath(DATASETS_PATH)):
        for filename in files:
            print("Reading %s ..." % filename)
            mesh_dataset, df_dataset, true_labels, true_types, true_cluster_indices = util.read_csv(
                os.path.join(root, filename))

            plot.plot_from_generator(mesh_dataset, true_labels, true_types, true_cluster_indices,
                                     os.path.join(RESULTS_PATH, filename + ".generator"))
            print("Calculating all clusterings for: %s" % filename)

            for algorithm, param_grid in CLUSTERERS.iteritems():
                util.grid_search(algorithm, param_grid, filename[:-4], mesh_dataset, df_dataset, true_labels,
                                 true_types, true_cluster_indices, RESULTS_PATH, n_jobs=4)


if __name__ == "__main__":
    main(sys.argv)
