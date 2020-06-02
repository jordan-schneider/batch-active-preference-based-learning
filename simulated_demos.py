from pathlib import Path
from typing import Tuple

import argh  # type: ignore
import numpy as np
from argh import arg
from numpy.random import random

from sampling import Sampler
from simulation_utils import create_env, run_algo


def get_simulated_feedback(
    simulation_object,
    input_A: np.ndarray,
    input_B: np.ndarray,
    reward: np.ndarray,
    fake_noise: bool,
) -> Tuple[np.ndarray, int]:
    simulation_object.feed(input_A)
    phi_A = simulation_object.get_features()
    simulation_object.feed(input_B)
    phi_B = simulation_object.get_features()
    normal = np.array(phi_A) - np.array(phi_B)
    preference = int(np.sign(np.dot(reward, normal)))

    if fake_noise:
        noise_prob = min(1, np.exp(preference * np.dot(reward, normal)))
        if random() > noise_prob:
            preference *= -1

    assert (len(normal.shape) == 1) and (normal.shape[0] == 4)
    assert (preference == -1) or (preference == 1)
    return normal, preference


@arg("--true-reward", nargs=4, type=float)
def batch(
    *,
    N: int = 100,
    M: int = 1000,
    b: int = 10,
    true_reward: Tuple[float, float, float, float] = random((4,)) * 2 - 1,
    outdir: Path = Path("questions"),
    fake_noise: bool = False,
):
    reward = np.array(true_reward)
    reward = reward / np.linalg.norm(reward)

    np.save(outdir / "reward", reward)

    if N % b != 0:
        print("N must be divisible to b")
        exit(0)
    B = 20 * b

    simulation_object = create_env("driver")
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    normals = []
    preferences = []
    inputs = []

    a_inputs = np.random.uniform(
        low=lower_input_bound,
        high=upper_input_bound,
        size=(b, simulation_object.feed_size),
    )
    b_inputs = np.random.uniform(
        low=lower_input_bound,
        high=upper_input_bound,
        size=(b, simulation_object.feed_size),
    )
    inputs.append((a_inputs, b_inputs))
    for input_a, input_b in zip(a_inputs, b_inputs):
        normal, preference = get_simulated_feedback(
            simulation_object, input_a, input_b, reward
        )
        normals.append(normal)
        preferences.append(preference)
    i = b
    while i < N:
        w_sampler.A = normals
        w_sampler.y = np.array(preferences).reshape(-1, 1)
        w_samples = w_sampler.sample(M)
        a_inputs, b_inputs = run_algo(
            "boundary_medoids", simulation_object, w_samples, b, B
        )
        inputs.append((a_inputs, b_inputs))
        for input_a, input_b in zip(a_inputs, b_inputs):
            normal, preference = get_simulated_feedback(
                simulation_object, input_a, input_b, reward
            )
            normals.append(normal)
            preferences.append(preference)
        i += b

    np.save(outdir / "normals", normals)
    np.save(outdir / "preferences", preferences)
    np.save(outdir / "inputs", inputs)


if __name__ == "__main__":
    argh.dispatch_command(batch)
