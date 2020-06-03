from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix  # type: ignore

from post import filter_halfplanes
from run_tests import assert_normals, find_reward_boundary, run_test


def test_lp():
    datadir = Path("questions/1")
    normals = np.load(datadir / "normals.npy")
    preferences = np.load(datadir / "preferences.npy")

    assert preferences.shape[0] > 0

    normals = (normals.T * preferences).T

    # normals = normals[:500]

    assert_normals(normals)

    normals_lp, _ = filter_halfplanes(
        normals=normals,
        epsilon=0.0,
        n_samples=1000,
        skip_remove_duplicates=True,
        skip_noise_filtering=True,
    )

    normals_epsilon, _ = filter_halfplanes(
        normals=normals,
        epsilon=0.0,
        n_samples=1000,
        skip_remove_duplicates=True,
        skip_noise_filtering=True,
        skip_redundancy_filtering=True,
    )

    rewards, aligned = find_reward_boundary(normals, n_rewards=10)

    epsilon_results = run_test(
        normals=normals_epsilon, fake_rewards=rewards, aligned=aligned
    )
    lp_results = run_test(normals=normals_lp, fake_rewards=rewards, aligned=aligned)

    assert np.all(epsilon_results == lp_results)
