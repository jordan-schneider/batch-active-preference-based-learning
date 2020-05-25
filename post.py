""" Post-process noise and consistency filtering. """

#%%

import logging
from pathlib import Path

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
    psi,
    s,
    n_samples,
    noise_threshold=0.7,
    epsilon=0,
    delta=0.05,
    logger=logging.getLogger(),
):
    simulation_object = create_env("driver")
    d = simulation_object.num_of_features

    # Filter halfspaces that are too noisy
    samples = sample(d, psi, s, n_samples)
    indices = (
        np.mean(np.dot(samples, psi.T) * s.reshape(-1) > 0, axis=0) > noise_threshold
    )
    psi = psi[indices]
    s = s[indices]

    print(f"indices.shape={indices.shape}")

    print(f"After noise filtering there are {psi.shape[0]} constraints left.")

    # Filter halfspaces that don't have 1-d probability that the expected return gap is epsilon.
    samples = sample(d, psi, s, n_samples)
    indices = (
        np.mean(np.dot(samples, psi.T) * s.reshape(-1) > epsilon, axis=0) > 1 - delta
    )
    psi = psi[indices]
    s = s[indices]

    print(f"After epsilon delta filtering there are {psi.shape[0]} constraints left.")

    # Remove redundant halfspaces
    psi, indices = remove_redundant_constraints(psi)
    psi = np.array(psi)
    s = s[indices]

    # These indices need to be applied to the inputs as well, or I need to calculate all of the
    # indices at once or something

    print(f"After removing redundancies there are {psi.shape[0]} constraints left.")

    return psi, s, indices


#%%


@arg("n-samples", type=int)
def main(
    n_samples: int,
    *,
    datadir: Path = Path("preferences"),
    epsilon: float = 0.0,
    delta: float = 0.05,
):
    psi, s, indices = filter_halfplanes(
        psi=np.load(datadir / "psi.npy"),
        s=np.load(datadir / "s.npy"),
        n_samples=n_samples,
        epsilon=epsilon,
        delta=delta,
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
    inputs = np.array([a_inputs, b_inputs])[:, indices]
    np.save(datadir / "filtered_inputs", indices)


if __name__ == "__main__":
    argh.dispatch_command(main)
