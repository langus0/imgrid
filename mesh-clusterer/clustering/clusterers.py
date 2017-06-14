# coding: utf-8
import copy
import os
from collections import Counter

from sklearn import cluster
from sklearn.neighbors import KDTree

import utils.clustering_util as util
import utils.plot as plot
from clustering.cube import MINORITY_CLASS, MAJORITY_CLASS
from clustering.cube_classifiers.cube_classifier import ONLY_MAJORITY
from clustering.cube_classifiers.probability_cube_classifier import SAFE, BORDERLINE, RARE, OUTLIER
from clustering.joiners.fast_joiner import FastJoiner
from clustering.mesh.fake_mesh import FakeMeshTypesPregenerated
from utils.clustering_util import get_label_from_probability

__author__ = "Mateusz Lango, Dariusz Brzezinski"


def k_means_clusterer(mesh_dataset, df_dataset, true_labels, k_clusters, n_init=10, max_iter=300, tol=1e-4,
                      random_state=None, draw_plots=False,
                      n_jobs=1):
    """
    K-means wrapper
    :param draw_plots:
    :param mesh_dataset: input mesh dataset (not used by this algorithm)
    :param df_dataset: input data frame
    :param true_labels: true classes of examples, needed for type counting
    :param k_clusters: number of expected clusters
    :param n_init: number of k-means re-runs
    :param max_iter: max iterations per run
    :param tol: convergence threshold
    :param random_state: random seed
    :param n_jobs: max jobs
    :return: cluster labels, number of clusters, number of safe, number of borderline, number of rare, number of outlier
    """
    k_means = cluster.KMeans(k_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state,
                             n_jobs=n_jobs)
    minority_df_dataset = util.filter_dataframe(df_dataset, true_labels)
    k_means.fit(minority_df_dataset)  # clustering made for MINORITY examples only
    labels, cluster_num, mesh, predicted_types = util.generate_clustering_stats(mesh_dataset, df_dataset, true_labels,
                                                                                k_means.labels_,
                                                                                cluster_centers=k_means.cluster_centers_)

    if draw_plots:
        plot.plot_mesh_clustering(mesh, util.to_file_name(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../results")),
            k_means_clusterer, {"k_clusters": k_clusters},
            df_dataset.name))
    return labels, cluster_num, Counter(predicted_types), mesh, k_means.labels_


def mesh_clusterer(mesh_dataset, df_dataset, true_labels, mesher, joiner, k_bins, alpha, option, dim_mod,
                   postprocessing, draw_plots):
    """
    Difficulty factor mesh clusterer.
    :param draw_plots:
    :param postprocessing:
    :param mesh_dataset: input mesh dataset
    :param df_dataset: input data frame (not used by this algorithm)
    :param true_labels: true classes of examples (not used by this algorithm)
    :param mesher: function that divides the attribute space into cells
    :param joiner: function that joins neighboring cells into clusters
    :param k_bins: number of bins to output by mesher
    :param alpha: significance level for join test used by the joiner
    :param option: join option used by joiner. One of: SIMPLE, STEEPEST, ALL_AT_ONCE.
    :param dim_mod: bin count modifier for datasets with more than one attribute. One of: "root", "divide".
    :return: cluster labels, number of clusters, number of safe, number of borderline, number of rare, number of outlier
    """
    mesh = mesher(mesh_dataset, k_bins, dim_mod)

    for joiner_instance in [FastJoiner(MINORITY_CLASS), FastJoiner(MAJORITY_CLASS), joiner(alpha, option)]:

        while True:
            # Uncomment to have step by step visualizations
            # plot_mesh(mesh)

            cube1, cube2 = joiner_instance.find_next_to_join(mesh)

            if cube1 is None or cube2 is None:
                break

            # this block is necessary for the all at once option: we can receive a merge request for a non-existing cube
            cube1 = mesh.get_cube(cube1.coordinates[0])
            cube2 = mesh.get_cube(cube2.coordinates[0])
            if cube1 == cube2:
                continue

            mesh.join_cubes(cube1, cube2)
            del cube2

    if draw_plots:
        plot.plot_mesh_clustering(mesh, util.to_file_name(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../results")),
            mesh_clusterer, {"mesher": mesher, "joiner": joiner, "alpha": alpha,
                             "option": option, "k_bins": k_bins, "dim_mod": dim_mod},
            df_dataset.name))

        plot.plot_mesh_clustering(postprocessing().postprocess(copy.deepcopy(mesh)), util.to_file_name(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../results")),
            mesh_clusterer, {"mesher": mesher, "joiner": joiner, "alpha": alpha,
                             "option": option, "k_bins": k_bins, "dim_mod": dim_mod,
                             'postprocessed': postprocessing.__name__},
            df_dataset.name), detailed=False)

    labels = mesh.generate_labels()
    cluster_num = postprocessing().postprocess(copy.deepcopy(mesh)).count_minority_cubes()
    types_list = Counter()
    for type_of_examples in [SAFE, BORDERLINE, RARE, OUTLIER]:
        types_list[type_of_examples] = mesh.count_examples_of_type(type_of_examples)
    predicted_types = 0
    return labels, cluster_num, types_list, mesh, predicted_types


def DBSCAN_clusterer(mesh_dataset, df_dataset, true_labels, eps, min_samples=3, random_state=None, draw_plots=False,
                     njobs=1):
    db_scan = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=njobs)
    minority_df_dataset = util.filter_dataframe(df_dataset, true_labels=true_labels)
    db_scan.fit(minority_df_dataset)
    clustering_labels = db_scan.labels_

    new_cluster_id = max(clustering_labels) + 1
    for i, label in enumerate(clustering_labels):
        if label == -1:
            clustering_labels[i] = new_cluster_id
            new_cluster_id += 1

    labels, cluster_num, mesh, predicted_types = util.generate_clustering_stats(mesh_dataset, df_dataset, true_labels,
                                                                                clustering_labels, eps=eps,
                                                                                dbscan=True)

    if draw_plots:
        plot.plot_mesh_clustering(mesh, util.to_file_name(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../results")),
            DBSCAN_clusterer, {"eps": eps, "min_samples": min_samples},
            df_dataset.name))
    return labels, cluster_num, Counter(predicted_types), mesh, db_scan.labels_


def napierala_clusterer(mesh_dataset, df_dataset, true_labels, k):
    """
    Categorizing examples by Napierala&Stefanowski method
    :param mesh_dataset: input mesh dataset
    :param df_dataset: input data frame (not used by this algorithm)
    :param true_labels: true classes of examples (not used by this algorithm)
    :param k: the size of neighbourhood
    :return:
    """
    predicted_types = detect_types_napierala(df_dataset, true_labels, k)
    return [0] * len(df_dataset), 1, Counter(predicted_types), FakeMeshTypesPregenerated(mesh_dataset,
                                                                                         predicted_types), 0


def detect_types_napierala(dataset, labels, k=5):
    kdt = KDTree(dataset, leaf_size=20, metric='euclidean')
    nn = kdt.query(dataset, k=k + 1, return_distance=False)
    types = list()
    for i in range(len(nn)):
        if labels[i] == MAJORITY_CLASS:
            types.append(ONLY_MAJORITY)
            continue
        knn = set(nn[i]) - {i}
        assert len(knn) == k
        knn_labels = Counter([labels[j] for j in knn])[MINORITY_CLASS]
        types.append(get_label_from_probability(float(knn_labels) / float(k), False))
    return types
