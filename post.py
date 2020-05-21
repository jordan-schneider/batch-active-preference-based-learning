""" Post-process noise and consistency filtering. """

#%%

# import argparse

import numpy as np

from sampling import Sampler
from simulation_utils import create_env, get_feedback, run_algo

#%%


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--psi",
        type=argparse.FileType("rb"),
        required=True,
        help="File containing psi values (difference in features vectors).",
    )
    parser.add_argument(
        "--s",
        type=argparse.FileType("rb"),
        required=True,
        help="File containing preferences.",
    )
    return parser.parse_args()


def inside(w: np.array, psi: np.array, s: int) -> bool:
    """ Determines if w is inside the constraint given by psi and s.
    psi = phi(trajectory_1) - phi(trajectory_2)
    w (s * (phi(trajectory_1) - phi(trajectory_2))) > 0
    so the constraint is true if sign (w psi) == s
    """
    return np.sign(np.dot(w, psi)) == s


def f(psi, s, n_samples, epsilon=0, delta=0.95):
    simulation_object = create_env("driver")
    d = simulation_object.num_of_features

    w_sampler = Sampler(d)
    w_sampler.A = psi
    w_sampler.y = s.reshape(-1, 1)
    samples = w_sampler.sample(n_samples)

    indices = np.dot(samples, psi.T) * s.reshape(-1) > -epsilon

    filtered = psi[np.mean(indices, axis=0) > delta]

    # TODO(joschnei) Calculate linear programming redundancies

    return filtered


#%%


def main():
    args = get_args()

    psi = np.load(args.psi)
    s = np.load(args.s).reshape(-1, 1)

    samples, indices = f(psi, s)


if __name__ == "__main__":
    main()
