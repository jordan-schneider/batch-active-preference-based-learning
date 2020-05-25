import gc

import numpy as np
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import pairwise_distances

import kmedoids


def get_y(psi_set, w_samples):
    return psi_set.dot(w_samples.T)


def get_term1(y):
    return np.sum(1.0 - np.exp(-np.maximum(y, 0)), axis=1)


def func_psi(psi_set, w_samples):

    y = get_y(psi_set, w_samples)

    gc.collect()
    term1 = get_term1(y)

    term2 = np.sum(1.0 - np.exp(-np.maximum(-y, 0)), axis=1)

    f = -np.minimum(term1, term2)

    return f


def select_top_candidates(simulation_object, w_samples, B, inputs_set, psi_set):

    d = simulation_object.num_of_features
    z = simulation_object.feed_size

    f_values = func_psi(psi_set, w_samples)

    id_input = np.argsort(f_values)

    inputs_set = inputs_set[id_input[0:B]]
    psi_set = psi_set[id_input[0:B]]

    return inputs_set, psi_set, z


def load_data(simulation_object):
    data = np.load("ctrl_samples/" + simulation_object.name + ".npz")
    inputs_set = data["inputs_set"].astype(np.half)
    psi_set = data["psi_set"].astype(np.half)
    return inputs_set, psi_set


def boundary_medoids(simulation_object, w_samples, b, B=200):
    w_samples = w_samples.astype(np.half)

    inputs_set, psi_set = load_data(simulation_object)

    inputs_set, psi_set, z = select_top_candidates(
        simulation_object, w_samples, B, inputs_set, psi_set
    )

    hull = ConvexHull(psi_set)

    simplices = np.unique(hull.simplices)

    boundary_psi = psi_set[simplices]
    boundary_inputs = inputs_set[simplices]
    D = pairwise_distances(boundary_psi, metric="euclidean")

    M, C = kmedoids.kMedoids(D, b)

    return boundary_inputs[M, :z], boundary_inputs[M, z:]
