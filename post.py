""" Post-process noise and consistency filtering. """

#%%

# import argparse

import logging
from pathlib import Path

import argh
import numpy as np
from argh import arg

from linear_programming import remove_redundant_constraints
from sampling import Sampler
from simulation_utils import create_env

#%%


def filter_halfplanes(
    psi, s, n_samples, epsilon=0, delta=0.95, logger=logging.getLogger()
):
    simulation_object = create_env("driver")
    d = simulation_object.num_of_features

    w_sampler = Sampler(d)
    w_sampler.A = psi
    w_sampler.y = s.reshape(-1, 1)
    samples = w_sampler.sample(n_samples)

    indices = np.dot(samples, psi.T) * s.reshape(-1) > -epsilon
    logger.debug(f"indices.shape={indices.shape}")

    filtered = psi[np.mean(indices, axis=0) > delta]
    logger.debug(f"epsilon-delta filtered.shape={filtered.shape}")

    filtered = remove_redundant_constraints(filtered)

    return filtered


#%%


@arg("n-samples", type=int)
def main(
    n_samples: int,
    *,
    psi: Path = Path("preferences/psi_set.npy"),
    s: Path = Path("preferences/s_set.npy"),
    out: Path = Path("preferences/filtered.npy"),
    epsilon: float = 0.0,
    delta: float = 0.95,
):
    filtered = filter_halfplanes(
        psi=np.load(psi),
        s=np.load(s),
        n_samples=n_samples,
        epsilon=epsilon,
        delta=delta,
    )
    np.save(out, filtered)


if __name__ == "__main__":
    argh.dispatch_command(main)
