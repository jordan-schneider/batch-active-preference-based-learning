""" Generate random pairs of trajectories as a baseline. """

from pathlib import Path

import argh  # type: ignore
import numpy as np

from simulated_demos import get_simulated_feedback
from simulation_utils import create_env


def main(n_trajectories: int = 0):
    reward = np.load(Path("questions/reward.npy"))
    simulation_object = create_env("driver")
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    a_inputs = np.random.uniform(
        low=lower_input_bound,
        high=upper_input_bound,
        size=(n_trajectories, simulation_object.feed_size),
    )
    b_inputs = np.random.uniform(
        low=lower_input_bound,
        high=upper_input_bound,
        size=(n_trajectories, simulation_object.feed_size),
    )

    normals = list()
    preferences = list()
    for a, b in zip(a_inputs, b_inputs):
        normal, preference = get_simulated_feedback(simulation_object, a, b, reward)
        normals.append(normal)
        preferences.append(preference)

    np.save(Path("questions/random_normals"), normals)
    np.save(Path("questions/random_preferences"), preferences)


if __name__ == "__main__":
    argh.dispatch_command(main)
