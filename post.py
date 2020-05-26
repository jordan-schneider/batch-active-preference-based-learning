""" Post-process noise and consistency filtering. """

#%%

import logging
from pathlib import Path
from typing import Optional

import argh
import numpy as np
from argh import arg

from linear_programming import remove_redundant_constraints
from sampling import Sampler
from simulation_utils import create_env

#%%


def sample(d, psi, s, n_samples):
    w_sampler = Sampler(d)
    w_sampler.A = psi
    w_sampler.y = s.reshape(-1, 1)
    return w_sampler.sample(n_samples)


def filter_halfplanes(
    psi, s, n_samples, noise_threshold=0.7, epsilon=0, delta=0.05, skip_lp: bool = True,
):
    simulation_object = create_env("driver")
    d = simulation_object.num_of_features

    # Filter halfspaces that are too noisy
    samples = sample(d, psi, s, n_samples)
    indices = (
        np.mean(np.dot(samples, psi.T) * s.reshape(-1) > 0, axis=0) > noise_threshold
    )

    print(f"After noise filtering there are {np.sum(indices)} constraints left.")

    # Filter halfspaces that don't have 1-d probability that the expected return gap is epsilon.
    filtered_psi = psi[indices]
    filtered_s = s[indices]
    samples = sample(d, filtered_psi, filtered_s, n_samples)

    tmp = (
        np.mean(
            np.dot(samples, filtered_psi.T) * filtered_s.reshape(-1) > epsilon, axis=0,
        )
        > 1 - delta
    )

    indices = np.where(indices)[0][tmp]
    filtered_psi = psi[indices]
    filtered_s = s[indices]

    print(f"After epsilon delta filtering there are {len(indices)} constraints left.")

    if not skip_lp:
        # Remove redundant halfspaces
        filtered_psi, constraint_indices = remove_redundant_constraints(psi[indices])

        constraint_indices = np.array(constraint_indices, dtype=np.int)

        indices = indices[constraint_indices]

        filtered_psi = np.array(filtered_psi)

        assert np.all(psi[indices] == filtered_psi)

        filtered_s = s[indices]

        print(f"After removing redundancies there are {len(indices)} constraints left.")

    return filtered_psi, filtered_s, indices


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
):
    psi = np.load(datadir / "psi.npy")
    s = np.load(datadir / "s.npy")
    if n_human_samples is not None:
        psi = psi[:n_human_samples]
        s = s[:n_human_samples]

    psi, s, indices = filter_halfplanes(
        psi=psi, s=s, n_samples=n_samples, epsilon=epsilon, delta=delta,
    )
    np.save(datadir / "filtered_psi", psi)
    np.save(datadir / "filtered_s", s)

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
