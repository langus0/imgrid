# coding: utf-8

import copy
import csv
import datetime as dt
import multiprocessing
import os
import shutil
import time
import traceback
import types
from collections import Counter
from collections import namedtuple

import pandas as pd
from scipy.spatial import distance as dist
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.grid_search import ParameterGrid
from sklearn.neighbors import KDTree

from clustering.cube import MINORITY_CLASS
from clustering.cube_classifiers.cube_classifier import SAFE, BORDERLINE, RARE, OUTLIER, ONLY_MAJORITY
from clustering.element import Element
from clustering.mesh.fake_mesh import FakeMesh
from clustering.metrics.f_score_types import ElementsTypeFScore
from clustering.metrics.g_mean import GMeanMetric
from clustering.metrics.precision_types import ElementsTypePrecision
from clustering.metrics.recall_types import ElementsTypeRecall

__author__ = "Mateusz Lango, Dariusz Brzezinski"

Clustering = namedtuple("Clustering",
                        "parameters labels cluster_num safe borderline rare outlier cluster_summaries ami")
lock = multiprocessing.Lock()


def read_csv(filename, delimiter=",", valid_numeric_attributes=None, class_attribute_num=-3, typology_attribute_num=-2,
             clusterIdx_attribute_num=-1):
    """
    Reads a dataset from a csv file.
    :param filename: file path of the dataset
    :param delimiter: csv file delimiter
    :param valid_numeric_attributes: valid column numbers of numerical attributes. If None, the function will treat all
    the columns as valid attributes, except for the class attribute specified by class_attribute_num.
    :param class_attribute_num: class attribute column index
    :param typology_attribute_num: typology attribute column index
    :param clusterIdx_attribute_num: cluster attribute column index
    :return: a list of Element objects
    """
    df_dataset = pd.read_csv(filename, sep=delimiter)
    class_labels = df_dataset.iloc[:, class_attribute_num].copy()
    typology_labels = df_dataset.iloc[:, typology_attribute_num].copy()
    cluster_indexes = df_dataset.iloc[:, clusterIdx_attribute_num].copy()
    region_labels = get_ground_truth(filename, delimiter, class_attribute_num)

    if valid_numeric_attributes is None:
        valid_numeric_attributes = range(df_dataset.shape[1])
        valid_numeric_attributes.remove(valid_numeric_attributes[class_attribute_num])
        valid_numeric_attributes.remove(valid_numeric_attributes[typology_attribute_num])
        valid_numeric_attributes.remove(valid_numeric_attributes[clusterIdx_attribute_num])

    df_dataset = df_dataset.iloc[:, valid_numeric_attributes].copy()
    df_dataset.name = os.path.basename(filename)
    mesh_dataset = []
    for i in df_dataset.index:
        if i % 1000 == 0:
            print "Dataset reading progress ", i
        mesh_dataset.append(Element(df_dataset.iloc[i, valid_numeric_attributes].tolist(), class_labels[i],
                                    typology_labels[i], cluster_indexes[i]))

    return mesh_dataset, df_dataset, region_labels, typology_labels, cluster_indexes


def get_ground_truth(filename, delimiter, class_attribute_num):
    df_dataset = pd.read_csv(filename, sep=delimiter)
    return df_dataset.iloc[:, class_attribute_num].copy()


def cleanup_and_prepare_folders(results_path):
    """
    Cleans folders and files with outdated results
    :param results_path: folder with clustering results
    """
    print("Cleaning up previous results...")

    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path)


def calculate_ami_without_outliers(mesh, postprocessing):
    if postprocessing is not None:
        mesh = postprocessing().postprocess(copy.deepcopy(mesh))
    predicted_labels, true_cluster_indices = [], []
    for index, cube in enumerate(mesh.get_cubes()):
        for element in cube.data:
            if element.clazz == MINORITY_CLASS and element.typeOfExampe != OUTLIER:
                predicted_labels.append(index)
                true_cluster_indices.append(element.clusterIdx)
    return metrics.adjusted_mutual_info_score(true_cluster_indices, predicted_labels)


def evaluate_clusters(algorithm_name, parameters, params, data, mesh_dataset, mesh, predicted_labels,
                      minority_predicted_labels, k_clusters, example_types_counter, true_labels, true_types,
                      true_cluster_indices, time_elapsed, results_path):
    # measure selected based on "Adjusting for Chance Clustering Comparison Measures"
    ami = calculate_ami_without_outliers(mesh, parameters.get('postprocessing', None))  # newer ami

    if "mesh" not in algorithm_name and "napierala" not in algorithm_name and len(set(minority_predicted_labels)) > 1:
        minority_dataframe = filter_dataframe(data, true_labels)

        notoutlier_clusters = set([i for i, j in Counter(minority_predicted_labels).iteritems() if j > 1])
        minority_dataframe = minority_dataframe.iloc[
            [i for i, j in enumerate(minority_predicted_labels) if j in notoutlier_clusters]]
        minority_predicted_labels = [i for i in minority_predicted_labels if i in notoutlier_clusters]

        if len(minority_dataframe.index) > 10000:
            minority_dataframe, _, minority_predicted_labels, _ = train_test_split(minority_dataframe,
                                                                                   minority_predicted_labels,
                                                                                   train_size=10000,
                                                                                   stratify=minority_predicted_labels)

        try:
            sc = metrics.silhouette_score(minority_dataframe, minority_predicted_labels)
        except:
            print 'Error', len(set(minority_predicted_labels)), Counter(minority_predicted_labels)
            sc = "err"
            # raise
    else:
        sc = "nan"

    cluster_summaries = ""
    [precision_safe, precision_border, precision_rare, precision_outlier], \
    [recall_safe, recall_border, recall_rare, recall_outlier], \
    f_score, g_mean = evaluate_types(algorithm_name, mesh, predicted_labels, true_labels, true_types)

    write_evaluation_to_file(algorithm_name, data.name, params, len(predicted_labels), k_clusters,
                             example_types_counter,
                             precision_safe, precision_border, precision_rare, precision_outlier,
                             recall_safe, recall_border, recall_rare, recall_outlier, f_score, g_mean, ami,
                             sc, time_elapsed, results_path)
    return cluster_summaries, ami


def evaluate_types(algorithm_name, mesh, predicted_labels, true_labels, true_types):
    precision = ElementsTypePrecision(mesh).calculate_precision_mesh_clusterer()
    recall = ElementsTypeRecall(mesh).calculate_recall_mesh_clusterer()
    f_score = ElementsTypeFScore(mesh).calculate_f_score_mesh_clusterer()
    pred, real = mesh.convert_to_lists()
    g_mean = GMeanMetric(pred, real).g_mean_score()
    return precision, recall, f_score, g_mean


def write_evaluation_to_file(algorithm_name, dataset_name, params, n, k_clusters, example_types_counter,
                             precision_safe, precision_border, precision_rare, precision_outlier,
                             recall_safe, recall_border, recall_rare, recall_outlier, f_score, g_mean, ami,
                             sc, time_elapsed,
                             results_path):
    """
    Appends an evaluation summary to a csv file.
    :param g_mean:
    :param recall_outlier:
    :param recall_rare:
    :param recall_border:
    :param recall_safe:
    :param precision_outlier:
    :param precision_rare:
    :param precision_border:
    :param precision_safe:
    :param example_types_counter:
    :param algorithm_name: name of the algorithm that produced the results being evaluated
    :param dataset_name: name of current data set
    :param params: algorithm parameters
    :param n: number of elements to cluster
    :param k_clusters: number of clusters
    :param f_score: calculated f score
    :param ami: adjusted mutual information score
    :param sc: Silhouette Score
    :param time_elapsed: time
    :param results_path: path to save results to
    :return:
    """
    file_path = os.path.join(results_path, "clustering_results.csv")

    if os.path.isfile(file_path):
        write_header = False
        mode = "ab"
    else:
        write_header = True
        mode = "wb"

    with open(file_path, mode) as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)

        if write_header:
            writer.writerow(
                ["Algorithm", "Data set name", "Params", "Examples", "No of clusters", "No safe", "No borderline",
                 "No rare", "No outlier", "Precision S", "Precision B", "Precision R", "Precision O", "Recall S",
                 "Recall B", "Recall R", "Recall O", "F-score", "G-mean", "Adjusted Mutual Information", "Silhouette",
                 "Time"])

        writer.writerow([algorithm_name, dataset_name, params, n, k_clusters, example_types_counter[SAFE],
                         example_types_counter[BORDERLINE], example_types_counter[RARE], example_types_counter[OUTLIER],
                         precision_safe, precision_border, precision_rare, precision_outlier, recall_safe,
                         recall_border, recall_rare, recall_outlier, f_score, g_mean, ami, sc, time_elapsed])


def parameters_to_string(parameters):
    """
    Converts algorithm parameters into a string format suitable for file names.
    :param parameters: parameter dictionary
    :return: string representation of algorithm parameters
    """
    result = "{"

    for p, v in parameters.iteritems():
        p_str = str(p)
        if isinstance(v, (type, types.ClassType)):
            v_str = v.__name__
        else:
            v_str = str(v)
        result += "%s=%s " % (p_str, v_str)

    result += "}"
    return result


def to_file_name(results_path, algorithm, parameters, dataset_name):
    return os.path.join(results_path, algorithm.__name__,
                        algorithm.__name__ + "_" + dataset_name + "_" + parameters_to_string(parameters))


def grid_search(algorithm, grid_params, dataset_name, mesh_dataset, df_dataset, true_labels, true_types,
                true_cluster_indices, results_path, n_jobs=1):
    print("%s: Parameter grid search for %s" % (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), algorithm.__name__))

    if not os.path.exists(os.path.join(results_path, algorithm.__name__)):
        os.makedirs(os.path.join(results_path, algorithm.__name__))
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(calculate_and_evaluate_clustering)(algorithm, parameters, dataset_name, mesh_dataset, df_dataset,
                                                          true_labels, true_types, true_cluster_indices, results_path)
        for parameters in ParameterGrid(grid_params))


def calculate_and_evaluate_clustering(algorithm, parameters, dataset_name, mesh_dataset, df_dataset, true_labels,
                                      true_types, true_cluster_indices, results_path):
    try:
        df_dataset.name = dataset_name
        starting_time = time.clock()
        labels, cluster_num, example_types_counter, mesh, minority_labels = algorithm(mesh_dataset, df_dataset,
                                                                                      true_labels, **parameters)
        time_elapsed = time.clock() - starting_time

        with lock:
            evaluate_clusters(algorithm.__name__, parameters, parameters_to_string(parameters),
                              df_dataset, mesh_dataset, mesh, labels, minority_labels,
                              cluster_num,
                              example_types_counter, true_labels, true_types,
                              true_cluster_indices, time_elapsed, results_path)

    except Exception as e:
        print(str(e))
        print traceback.print_exc()


def translate_clustering_labels_to_types(labels, true_classes, only_minority=True):
    types_of_clusters = list()
    predicted_types = list()
    for cluster in range(0, max(labels) + 1):
        sum_in_cluster = 0
        sum_of_minority_class_examples = 0
        for index, element in enumerate(labels):
            if element == cluster:
                sum_in_cluster += 1
                if true_classes[index] == MINORITY_CLASS:
                    sum_of_minority_class_examples += 1
        if sum_in_cluster == 0:
            types_of_clusters.append(OUTLIER)
        else:
            prop = float(sum_of_minority_class_examples) / sum_in_cluster
            types_of_clusters.append(get_label_from_probability(prop))
    if only_minority:
        for index, element in enumerate(labels):
            if true_classes[index] == MINORITY_CLASS:
                if element == -1:
                    predicted_types.append(OUTLIER)
                else:
                    predicted_types.append(types_of_clusters[element])
    else:
        for index, element in enumerate(labels):
            if true_classes[index] == MINORITY_CLASS:
                if element == -1:
                    predicted_types.append(OUTLIER)
                else:
                    predicted_types.append(types_of_clusters[element])
            else:
                predicted_types.append("MAJ")
    return predicted_types


def take_minority_examples_types(true_labels, true_types):
    return [true_types[index] for index, iterator in enumerate(true_labels) if iterator == MINORITY_CLASS]


def filter_dataframe(df_dataset, true_labels):
    return df_dataset[true_labels == MINORITY_CLASS]


def cluster_all(df_dataset, true_labels, minority_clusters, cluster_centers=None, dbscan=None, eps=None):
    if dbscan:
        assert eps is not None
        return cluster_all_dbscan(df_dataset, true_labels, minority_clusters, eps)
    else:
        assert cluster_centers is not None
        return cluster_all_kmeans(df_dataset, true_labels, minority_clusters, cluster_centers)


def cluster_all_kmeans(df_dataset, true_labels, minority_clusters, cluster_centers):
    labels = list()
    min_cl_iterator = 0  # min_cl_iterator iterates in minority_clusters to append to labels list
    min_dist = 0
    cluster = 0
    for i in range(len(true_labels)):
        if true_labels[i] == MINORITY_CLASS:
            labels.append(minority_clusters[min_cl_iterator])
            min_cl_iterator += 1
        else:  # example is from MAJORITY CLASS
            for j in range(len(cluster_centers)):
                distance = dist.euclidean(list(df_dataset.iloc[i, :]), list(cluster_centers[j]))
                if j == 0:
                    min_dist = distance
                    cluster = 0
                else:
                    if distance < min_dist:
                        cluster = j
                        min_dist = distance
            labels.append(cluster)
    return labels


def cluster_all_dbscan(df_dataset, true_labels, minority_clusters, eps):
    labels = list()
    min_cl_iterator = 0  # min_cl_iterator iterates in minority_clusters to append to labels list
    # min_cl_iterator2 does the same job, but for dbscan cluster number
    majority_cluster_id = max(minority_clusters) + 1
    kdt = KDTree(filter_dataframe(df_dataset, true_labels=true_labels), leaf_size=20, metric='euclidean')
    for i in range(len(true_labels)):
        if true_labels[i] == MINORITY_CLASS:
            labels.append(minority_clusters[min_cl_iterator])
            min_cl_iterator += 1
        else:  # example is from MAJORITY CLASS
            distance, idx = kdt.query([df_dataset.ix[i]], k=1, return_distance=True)
            if distance[0][0] <= eps:
                labels.append(minority_clusters[idx[0][0]])
            else:
                labels.append(majority_cluster_id)
    return labels


def generate_clustering_stats(mesh_dataset, df_dataset, true_labels, clustering_labels, cluster_centers=None,
                              dbscan=False, eps=None):
    labels = cluster_all(df_dataset, true_labels, clustering_labels, cluster_centers, dbscan, eps)  # majority added
    cluster_num = len(set(labels))
    predicted_types = translate_clustering_labels_to_types(labels, true_labels)
    mesh = FakeMesh(mesh_dataset, labels)  # here update results2mesh
    return labels, cluster_num, mesh, predicted_types


def get_label_from_probability(prob, only_majority=True):
    if prob > 0.7:
        return SAFE
    elif prob > 0.3:
        return BORDERLINE
    elif prob > 0.1:
        return RARE
    elif prob > 0:
        return OUTLIER
    else:
        if only_majority:
            return ONLY_MAJORITY
        else:
            return OUTLIER
