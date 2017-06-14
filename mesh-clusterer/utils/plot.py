# coding: utf-8

import matplotlib.patches as patches
import matplotlib.pyplot as plt

import clustering_util as util
from clustering.cube import MINORITY_CLASS
from clustering.cube_classifiers.probability_cube_classifier import SAFE, BORDERLINE, RARE, OUTLIER, \
    ProbabilityCubeClassifier
from clustering.mesh.fake_mesh import FakeMesh

__author__ = "Mateusz Lango, Dariusz Brzezinski"


def plot_types(dataset, labels, types, save_to_file=None, idx=0, idy=1):
    fig1 = plt.figure()
    fig1.patch.set_facecolor('white')

    for index, elem in enumerate(types):
        if labels[index] == MINORITY_CLASS:
            symbol = 'o'
            if elem == SAFE:
                color = "#4daf41"
            elif elem == BORDERLINE:
                color = "#ff7f00"
            elif elem == RARE:
                color = "#e41a1c"
            elif elem == OUTLIER:
                color = "black"
            else:
                color = "black"
        else:
            symbol = ","
            color = "black"
        plt.plot(dataset[index].data[idx], dataset[index].data[idy], symbol, color=color)
    if save_to_file is None:
        plt.show()
    else:
        plt.savefig(save_to_file + "_types.png", bbox_inches='tight')
    plt.close(fig1)


def plot_clusters(dataset, labels, clusters, save_to_file=None,
                  markers=('o', 'v', '*', 'D', '+', 's', '^', 'x', '<', 'h', '>'),
                  colors=("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff23", "#a65628", "#f781bf", "#999999")
                  , idx=0, idy=1):
    fig1 = plt.figure()
    fig1.patch.set_facecolor('white')
    clusters = list(clusters)

    for index, elem in enumerate(labels):
        if elem == MINORITY_CLASS:
            symbol = markers[clusters[index] // len(colors)]
            color = colors[clusters[index] % len(colors)]
        else:
            symbol = ","
            color = "black"
        plt.plot(dataset[index].data[idx], dataset[index].data[idy], symbol, color=color)
    if save_to_file is None:
        plt.show()
    else:
        plt.savefig(save_to_file + "_clusters.png", bbox_inches='tight')
    plt.close(fig1)
    plt.close()


def plot_mesh(mesh, save_to_file=None, alpha=0.5, patterns=('', '///', 'x', '\\\\', '-', '+', 'o', 'O', '.', '*'),
              colors=(
              "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff23", "#a65628", "#f781bf", "#999999")):
    if isinstance(mesh, FakeMesh):
        print("WARN: FakeMesh - plotting ommited")
        return
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    c_idx = 0
    p_idx = 0

    for cube in mesh.get_cubes():

        for element in cube.data:
            symbol = 'o' if element.clazz == MINORITY_CLASS else ","
            color = "black"
            plt.plot(element.data[0], element.data[1], symbol, color=color)

        for coords in cube.coordinates:
            width, height = mesh.get_coord_width(coords)
            ax1.add_patch(
                patches.Rectangle(
                    tuple(mesh.convert_to_true_coordinates(coords)),  # (x,y)
                    width,
                    height,
                    alpha=alpha,
                    facecolor=colors[c_idx], hatch=patterns[p_idx]
                )
            )
        c_idx += 1
        if c_idx >= len(colors):
            c_idx = 0
            p_idx += 1
            if p_idx >= len(patterns):
                print("WARN: Too much cubes...")
                p_idx = 0
    if save_to_file is None:
        plt.show()
    else:
        plt.savefig(save_to_file + "mesh.png", bbox_inches='tight')
    plt.close(fig1)
    plt.close()


def plot_from_generator(mesh_dataset, true_labels, true_types, true_cluster_indices, save_to_file=None, idx=0, idy=1):
    plot_types(mesh_dataset, true_labels, true_types, save_to_file, idx=idx, idy=idy)
    plot_clusters(mesh_dataset, true_labels, true_cluster_indices, save_to_file, idx=idx, idy=idy)


def plot_mesh_clustering(mesh, save_to_file=None, detailed=True):
    plot_mesh(mesh, save_to_file)

    mesh_dataset = []
    labels = []
    clusters = []
    types = []

    for idx, cube in enumerate(mesh.get_cubes()):
        type_of_cube = ProbabilityCubeClassifier().class_of_cube(cube)[0]

        for element in cube.data:
            labels.append(element.clazz)
            types.append(type_of_cube)
            clusters.append(idx)
            mesh_dataset.append(element)
    plot_clusters(mesh_dataset, labels, clusters, save_to_file)

    if detailed:
        plot_types(mesh_dataset, labels, types, save_to_file)


def plot_clusterings(mesh_dataset, true_classes, true_types, labels, save_to_file=None):
    predicted_types = util.translate_clustering_labels_to_types(labels, true_classes, False)
    plot_types(mesh_dataset, true_classes, predicted_types, save_to_file)
    plot_clusters(mesh_dataset, true_classes, labels, save_to_file)
