import sys
from pathlib import Path
from typing import List, Tuple

import fire
import numpy as np

import algos
from sampling import Sampler
from simulation_utils import create_env, get_feedback, run_algo


def elicit_preferences(
    a_inputs, b_inputs, simulation_object, normals, preferences, outdir
):
    for input_a, input_b in zip(a_inputs, b_inputs):
        normal, preference = get_feedback(simulation_object, input_a, input_b)
        normals.append(normal)
        preferences.append(preference)
        np.save(outdir / "normals", normals)
        np.save(outdir / "preferences", preferences)
    return normals, preferences


def update_inputs(a_inputs, b_inputs, inputs, outdir):
    """Adds a new pair of input trajectories (a_inputs, b_inputs) to the inputs list and saves it."""
    inputs.append((a_inputs, b_inputs))
    np.save(outdir / "inputs", inputs)


def sample_inputs(normals, preferences, w_sampler, simulation_object, M, b, B):
    w_samples = sample_weights(normals, preferences, w_sampler, M)
    a_inputs, b_inputs = run_algo(
        "boundary_medoids", simulation_object, w_samples, b, B
    )
    return a_inputs, b_inputs


def sample_weights(normals, preferences, w_sampler, M):
    w_sampler.A = normals
    w_sampler.y = np.array(preferences).reshape(-1, 1)
    w_samples = w_sampler.sample(M)
    return w_samples


def save_reward(normals, preferences, w_sampler, M, outdir):
    w_samples = sample_weights(normals, preferences, w_sampler, M)
    mean_weight = np.mean(w_samples, axis=0)
    normalized_mean_weight = mean_weight / np.linalg.norm(mean_weight)
    np.save(outdir / "reward.npy", normalized_mean_weight)


def batch(
    N: int, M: int, b: int, outdir: Path = Path("questions"),
):
    print("Space to start/stop the video, escape to return.")

    outdir = Path(outdir)
    if N % b != 0:
        print("N must be divisible to b")
        exit(0)
    B = 20 * b

    simulation_object = create_env("driver")
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    normals: List[np.ndarray] = []
    preferences: List[np.ndarray] = []

    inputs: List[Tuple[np.ndarray, np.ndarray]] = []

    try:
        for i in range(0, N, b):
            if i == 0:
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
            else:
                print("Thinking...")
                a_inputs, b_inputs = sample_inputs(
                    normals, preferences, w_sampler, simulation_object, M, b, B
                )
            update_inputs(a_inputs, b_inputs, inputs, outdir)

            normals, preferences = elicit_preferences(
                a_inputs, b_inputs, simulation_object, normals, preferences, outdir
            )
    except KeyboardInterrupt:
        # Pass through to finally
        print("Saving results, please do not exit again.")
    finally:
        save_reward(normals, preferences, w_sampler, M, outdir)


if __name__ == "__main__":
    fire.Fire(batch)
