""" Runs the test set generated by post.py by generating fake reward weights and seeing how many
are caught by the preferences."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argh  # type: ignore
import numpy as np
from argh import arg
from numpy.linalg import norm
from scipy.stats import multivariate_normal  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore

from post import filter_halfplanes

N_FEATURES = 4


def assert_normals(normals: np.ndarray) -> None:
    """ Asserts the given array is an array of normal vectors defining half space constraints."""
    shape = normals.shape
    assert len(shape) == 2
    assert shape[1] == N_FEATURES


def assert_reward(reward: np.ndarray) -> None:
    """ Asserts the given array is might be a reward feature vector. """
    assert reward.shape == (N_FEATURES,)
    assert abs(norm(reward) - 1) < 0.000001


def normalize(vectors: np.ndarray) -> np.ndarray:
    """ Takes in a 2d array of row vectors and ensures each row vector has an L_2 norm of 1."""
    return (vectors.T / norm(vectors, axis=1)).T


def make_rewards(n_rewards: int) -> np.ndarray:
    """ Makes n_rewards uniformly sampled reward vectors of unit length."""
    assert n_rewards > 0
    dist = multivariate_normal(mean=np.zeros(N_FEATURES))
    rewards = normalize(dist.rvs(size=n_rewards))
    return rewards


def find_reward_boundary(
    normals: np.ndarray, n_rewards: int, reward: np.ndarray, epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Generates n_rewards reward vectors and determines which are aligned. """
    assert_normals(normals)
    assert n_rewards > 0
    assert epsilon >= 0.0
    assert_reward(reward)

    rewards = make_rewards(n_rewards)

    normals = normals[np.dot(reward, normals.T) > epsilon]

    ground_truth_alignment = np.all(np.dot(rewards, normals.T) > 0, axis=1)

    assert ground_truth_alignment.shape == (n_rewards,)
    assert rewards.shape == (n_rewards, N_FEATURES)

    return rewards, ground_truth_alignment


def run_test(normals: np.ndarray, fake_rewards: np.ndarray) -> np.ndarray:
    """ Returns the predicted alignment of the fake rewards by the normals. """
    assert_normals(normals)
    results = np.all(np.dot(fake_rewards, normals.T) > 0, axis=1)
    return results


def eval_test(
    normals: np.ndarray, fake_rewards: np.ndarray, aligned: np.ndarray,
) -> np.ndarray:
    """ Makes a confusion matrix by evaluating a test on the fake rewards. """
    assert fake_rewards.shape[0] == aligned.shape[0]

    for fake_reward in fake_rewards:
        assert_reward(fake_reward)

    if normals.shape[0] > 0:
        results = run_test(normals, fake_rewards)
        print(
            f"predicted true={np.sum(results)}, predicted false={results.shape[0] - np.sum(results)}"
        )
        return confusion_matrix(y_true=aligned, y_pred=results, labels=[False, True])
    else:
        return confusion_matrix(
            y_true=aligned,
            y_pred=np.ones(aligned.shape, dtype=bool),
            labels=[False, True],
        )


def make_outname(
    skip_remove_duplicates: bool,
    skip_noise_filtering: bool,
    skip_epsilon_filtering: bool,
    skip_redundancy_filtering: bool,
    base: str = "out",
) -> str:
    outname = base
    if skip_remove_duplicates:
        outname += ".skip_duplicates"
    if skip_noise_filtering:
        outname += ".skip_noise"
    if skip_epsilon_filtering:
        outname += ".skip_epsilon"
    if skip_redundancy_filtering:
        outname += ".skip_lp"
    outname += ".pkl"
    return outname


@arg("--epsilons", nargs="+", type=float)
@arg("--human-samples", nargs="+", type=int)
def gt(
    epsilons: List[float] = [0.0],
    n_rewards: int = 100,
    human_samples: List[int] = [1],
    n_model_samples: int = 1000,
    normals_name: Path = Path("normals.npy"),
    preferences_name: Path = Path("preferences.npy"),
    true_reward_name: Path = Path("true_reward.npy"),
    datadir: Path = Path("questions"),
    skip_remove_duplicates: bool = False,
    skip_noise_filtering: bool = False,
    skip_epsilon_filtering: bool = False,
    skip_redundancy_filtering: bool = False,
    replications: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """ Run tests with full data to determine how much reward noise gets"""
    if replications is not None:
        start, stop = replications.split("-")
        for replication in range(int(start), int(stop) + 1):
            gt(
                epsilons,
                n_rewards,
                human_samples,
                n_model_samples,
                normals_name,
                preferences_name,
                datadir / str(replication),
                skip_remove_duplicates,
                skip_noise_filtering,
                skip_epsilon_filtering,
                skip_redundancy_filtering,
                overwrite=overwrite,
            )
        return

    normals = np.load(datadir / normals_name)
    preferences = np.load(datadir / preferences_name)
    reward = np.load(datadir / true_reward_name)

    assert preferences.shape[0] > 0
    assert normals.shape[0] > 0
    assert reward.shape == (N_FEATURES,)

    normals = (normals.T * preferences).T
    assert_normals(normals)

    confusion_path = datadir / make_outname(
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="confusion",
    )
    test_path = datadir / make_outname(
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="indices",
    )

    results: Dict[Tuple[float, int], np.ndarray]
    results = (
        pickle.load(open(confusion_path, "rb"))
        if confusion_path.exists() and not overwrite
        else dict()
    )

    minimal_tests: Dict[Tuple[float, int], np.ndarray]
    minimal_tests = (
        pickle.load(open(test_path, "rb"))
        if test_path.exists() and not overwrite
        else dict()
    )

    for epsilon in epsilons:
        if not overwrite and all(
            [(epsilon, n) in results.keys() for n in human_samples]
        ):
            continue

        rewards, aligned = find_reward_boundary(normals, n_rewards, reward, epsilon)
        print(
            f"aligned={np.sum(aligned)}, unaligned={aligned.shape[0] - np.sum(aligned)}"
        )
        for n in human_samples:
            if not overwrite and (epsilon, n) in results.keys():
                continue

            print(f"Working on epsilon={epsilon}, n={n}")
            filtered_normals = normals[:n]
            filtered_normals, indices = filter_halfplanes(
                normals=filtered_normals,
                n_samples=n_model_samples,
                epsilon=epsilon,
                skip_remove_duplicates=skip_remove_duplicates,
                skip_noise_filtering=skip_noise_filtering,
                skip_epsilon_filtering=skip_epsilon_filtering,
                skip_redundancy_filtering=skip_redundancy_filtering,
            )

            minimal_tests[(epsilon, n)] = indices

            confusion = eval_test(
                normals=filtered_normals, fake_rewards=rewards, aligned=aligned,
            )

            assert confusion.shape == (2, 2)

            results[(epsilon, n)] = confusion

    pickle.dump(results, open(confusion_path, "wb"))
    pickle.dump(minimal_tests, open(test_path, "wb"))


@arg("--epsilons", nargs="+", type=float)
@arg("--human-samples", nargs="+", type=int)
def human(
    epsilons: List[float] = [0.0],
    n_rewards: int = 100,
    human_samples: List[int] = [1],
    n_model_samples: int = 1000,
    normals_name: Path = Path("normals.npy"),
    preferences_name: Path = Path("preferences.npy"),
    datadir: Path = Path("questions"),
    skip_remove_duplicates: bool = False,
    skip_noise_filtering: bool = False,
    skip_epsilon_filtering: bool = False,
    skip_redundancy_filtering: bool = False,
    overwrite: bool = False,
):
    normals = np.load(datadir / normals_name)
    preferences = np.load(datadir / preferences_name)
    assert preferences.shape[0] > 0

    normals = (normals.T * preferences).T
    assert_normals(normals)

    test_path = datadir / make_outname(
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="indices",
    )
    test_results_path = datadir / make_outname(
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="test_results",
    )

    minimal_tests: Dict[Tuple[float, int], np.ndarray]
    minimal_tests = (
        pickle.load(open(test_path, "rb"))
        if test_path.exists() and not overwrite
        else dict()
    )
    results: Dict[Tuple[float, int], np.ndarray]
    results = (
        pickle.load(open(test_results_path, "rb"))
        if test_results_path.exists() and not overwrite
        else dict()
    )

    fake_rewards = make_rewards(n_rewards)

    for epsilon in epsilons:
        for n in human_samples:
            if not overwrite and (epsilon, n) in minimal_tests.keys():
                continue

            print(f"Working on epsilon={epsilon}, n={n}")
            filtered_normals = normals[:n]
            filtered_normals, indices = filter_halfplanes(
                normals=filtered_normals,
                n_samples=n_model_samples,
                epsilon=epsilon,
                skip_remove_duplicates=skip_remove_duplicates,
                skip_noise_filtering=skip_noise_filtering,
                skip_epsilon_filtering=skip_epsilon_filtering,
                skip_redundancy_filtering=skip_redundancy_filtering,
            )

            minimal_tests[(epsilon, n)] = indices

            results[(epsilon, n)] = run_test(filtered_normals, fake_rewards)

    pickle.dump(minimal_tests, open(test_path, "wb"))
    pickle.dump(results, open(test_results_path, "wb"))


if __name__ == "__main__":
    argh.dispatch_commands([gt, human])
