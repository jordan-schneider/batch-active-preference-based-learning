from typing import Tuple

import numpy as np


def remove_equiv(preferences: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """ Finds equivalence preferences and removes them + the associated elements of *arrays. """
    indices = preferences != 0
    preferences = preferences[indices]
    out_arrays = list()
    for array in arrays:
        out_arrays.append(array[indices])
    return (preferences, *out_arrays)


def add_equiv_constraints(
    preferences: np.ndarray, normals: np.ndarray, equiv_prob: float
) -> np.ndarray:
    """ Adds equivalence constraints to a set of halspace constraints. """
    out_normals = list()
    for preference, normal in zip(preferences, normals):
        if preference == 0:
            max_return_diff = equiv_prob - np.log(2 * equiv_prob - 2)
            # w phi >= -max_return_diff
            # w phi + max_reutrn_diff >=0
            # w phi <= max_return diff
            # 0 <= max_return_diff - w phi
            out_normals.append(np.append(normal, [max_return_diff]))
            out_normals.append(np.append(-normals, [max_return_diff]))
        elif preference == 1 or preference == -1:
            out_normals.append(np.append(normal * preference, [0]))

    return np.ndarray(out_normals)
