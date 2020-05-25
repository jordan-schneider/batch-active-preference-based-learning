import sys
from pathlib import Path
from typing import Tuple

import argh
import numpy as np
from argh import arg

from sampling import Sampler
from simulation_utils import create_env, run_algo


def get_simulated_feedback(simulation_object, input_A, input_B, reward):
    simulation_object.feed(input_A)
    phi_A = simulation_object.get_features()
    simulation_object.feed(input_B)
    phi_B = simulation_object.get_features()
    psi = np.array(phi_A) - np.array(phi_B)
    s = np.sign(np.dot(reward, psi))
    return psi, s


@arg("--true-reward", nargs=4, type=float)
def batch(
    *,
    N: int = 100,
    M: int = 1000,
    b: int = 10,
    true_reward: Tuple[float, float, float, float] = np.random.random((4,)) * 2 - 1,
    outdir: Path = Path("preferences")
):
    reward = np.array(true_reward)
    reward = reward / np.linalg.norm(reward)
    if N % b != 0:
        print("N must be divisible to b")
        exit(0)
    B = 20 * b

    simulation_object = create_env("driver")
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
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
        psi, s = get_simulated_feedback(simulation_object, input_a, input_b, reward)
        psi_set.append(psi)
        s_set.append(s)
    i = b
    while i < N:
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1, 1)
        w_samples = w_sampler.sample(M)
        a_inputs, b_inputs = run_algo(
            "boundary_medoids", simulation_object, w_samples, b, B
        )
        inputs.append((a_inputs, b_inputs))
        for input_a, input_b in zip(a_inputs, b_inputs):
            psi, s = get_simulated_feedback(simulation_object, input_a, input_b, reward)
            psi_set.append(psi)
            s_set.append(s)
        i += b

    np.save(outdir / "psi", psi_set)
    np.save(outdir / "s", s_set)
    np.save(outdir / "inputs", inputs)


if __name__ == "__main__":
    argh.dispatch_command(batch)
