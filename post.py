""" Post-process noise and consistency filtering. """

#%%

from pathlib import Path
from typing import List, Optional, Tuple

import argh  # type: ignore
import numpy as np
from argh import arg
from scipy.spatial import distance  # type: ignore

from linear_programming import remove_redundant_constraints
from sampling import Sampler
from simulation_utils import create_env

#%%


def sample(
    reward_dimension: int, normals: np.ndarray, preferences: np.ndarray, n_samples: int,
) -> np.ndarray:
    """ Samples n_samples rewards via MCMC. """
    w_sampler = Sampler(reward_dimension)
    w_sampler.A = normals
    w_sampler.y = preferences.reshape(-1, 1)
    return w_sampler.sample(n_samples)


def remove_duplicates(normals: np.ndarray, precision=0.0001) -> List[np.ndarray]:
    """ Remove halfspaces that have small cosine similarity to another. """
    out: List[np.ndarray] = []
    for normal in normals:
        for accepted_normal in out:
            if distance.cosine(normal, accepted_normal) < precision:
                break
        out.append(normal)
    return out


def filter_halfplanes(
    normals: np.ndarray,
    preferences: np.ndarray,
    n_samples: int,
    noise_threshold: float = 0.7,
    epsilon: float = 0.0,
    delta: float = 0.05,
    skip_lp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Filters test questions by removing noise answers, requiring answers have a gap of at 
    least epsilon, and removing redundant questions via linear programming. """
    reward_dimension = create_env("driver").num_of_features

    # Filter halfspaces that are too noisy
    rewards = sample(
        reward_dimension=reward_dimension,
        normals=normals,
        preferences=preferences,
        n_samples=n_samples,
    )
    indices = (
        np.mean(np.dot(rewards, normals.T) * preferences.reshape(-1) > 0, axis=0)
        > noise_threshold
    )

    print(f"After noise filtering there are {np.sum(indices)} constraints left.")

    # Filter halfspaces that don't have 1-d probability that the expected return gap is epsilon.
    filtered_normals = normals[indices]
    filtered_preferences = preferences[indices]
    rewards = sample(
        reward_dimension=reward_dimension,
        normals=normals,
        preferences=preferences,
        n_samples=n_samples,
    )

    filtered_indices = (
        np.mean(
            np.dot(rewards, filtered_normals.T) * filtered_preferences.reshape(-1)
            > epsilon,
            axis=0,
        )
        > 1 - delta
    )

    indices = np.where(indices)[0][filtered_indices]
    filtered_normals = normals[indices]
    filtered_preferences = preferences[indices]

    print(f"After epsilon delta filtering there are {len(indices)} constraints left.")

    if not skip_lp:
        # Remove redundant halfspaces
        filtered_normals, constraint_indices = remove_redundant_constraints(
            remove_duplicates(normals[indices])
        )

        constraint_indices = np.array(constraint_indices, dtype=np.int)

        indices = indices[constraint_indices]

        filtered_normals = np.array(filtered_normals)

        assert np.all(normals[indices] == filtered_normals)

        filtered_preferences = preferences[indices]

        print(f"After removing redundancies there are {len(indices)} constraints left.")

    return filtered_normals, filtered_preferences, indices


#%%


@arg("n-samples", type=int)
@arg("--n-human-samples", type=int)
def main(
    n_samples: int,
    *,
    datadir: Path = Path("preferences"),
    epsilon: float = 0.0,
    delta: float = 0.05,
    n_human_samples: Optional[int] = None,
) -> None:
    normals = np.load(datadir / "psi.npy")
    preferences = np.load(datadir / "s.npy")
    if n_human_samples is not None:
        normals = normals[:n_human_samples]
        preferences = preferences[:n_human_samples]

    normals, preferences, indices = filter_halfplanes(
        normals=normals,
        preferences=preferences,
        n_samples=n_samples,
        epsilon=epsilon,
        delta=delta,
    )
    np.save(datadir / "filtered_psi", normals)
    np.save(datadir / "filtered_s", preferences)

    inputs = np.load(datadir / "inputs.npy")
    a_inputs = inputs[:, 0, :, :].reshape(
        inputs.shape[0] * inputs.shape[2], inputs.shape[3]
    )
    b_inputs = inputs[:, 1, :, :].reshape(
        inputs.shape[0] * inputs.shape[2], inputs.shape[3]
    )
    inputs = np.array([a_inputs, b_inputs])
    if n_human_samples is not None:
        inputs = inputs[:, :n_human_samples]
    inputs = inputs[:, indices]
    np.save(datadir / "filtered_inputs", inputs)


if __name__ == "__main__":
    argh.dispatch_command(main)
