import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import argh  # type: ignore
import numpy as np
from argh import arg
from numpy.random import random

import algos
from models import Driver
from sampling import Sampler
from simulation_utils import create_env, get_feedback, run_algo

GetPrefFunc = Callable[[Driver, np.ndarray, np.ndarray], Tuple[np.ndarray, int]]


def append(a: Optional[np.ndarray], b: Union[np.ndarray, int]) -> np.ndarray:
    if a is None:
        if isinstance(b, np.ndarray):
            return b.reshape(1, *b.shape)
        elif isinstance(b, int):
            return np.ndarray(b).reshape((1,))
    else:
        if isinstance(b, np.ndarray):
            return np.append(a, b.reshape((1, *b.shape)), axis=0)
        elif isinstance(b, int):
            return np.append(a, b)


def elicit_preferences(
    a_inputs: np.ndarray,
    b_inputs: np.ndarray,
    simulation_object: Driver,
    normals: Optional[np.ndarray],
    preferences: Optional[np.ndarray],
    outdir: Path,
    get_preference: GetPrefFunc,
) -> Tuple[np.ndarray, np.ndarray]:
    for input_a, input_b in zip(a_inputs, b_inputs):
        normal, preference = get_preference(simulation_object, input_a, input_b)
        normals = append(normals, normal)
        preferences = append(preferences, preference)
        np.save(outdir / "normals.npy", normals)
        np.save(outdir / "preferences.npy", preferences)
    return normals, preferences


def update_inputs(
    a_inputs: np.ndarray, b_inputs: np.ndarray, inputs: np.ndarray, outdir: Path,
) -> np.ndarray:
    """Adds a new pair of input trajectories (a_inputs, b_inputs) to the inputs list and saves it."""
    inputs = append(inputs, np.stack([a_inputs, b_inputs]))
    np.save(outdir / "inputs.npy", inputs)
    return inputs


def sample_inputs(
    normals: np.ndarray,
    preferences: np.ndarray,
    w_sampler,
    simulation_object: Driver,
    M: int,
    b: int,
    B: int,
) -> Tuple[np.ndarray, np.ndarray]:
    w_samples = sample_weights(normals, preferences, w_sampler, M)
    a_inputs, b_inputs = run_algo(
        "boundary_medoids", simulation_object, w_samples, b, B
    )
    return a_inputs, b_inputs


def sample_weights(
    normals: np.ndarray, preferences: np.ndarray, w_sampler, M: int
) -> np.ndarray:
    w_sampler.A = normals
    w_sampler.y = preferences.reshape(-1, 1)
    w_samples = w_sampler.sample(M)
    return w_samples


def save_reward(
    normals: np.ndarray, preferences: np.ndarray, w_sampler, M: int, outdir: Path
):
    w_samples = sample_weights(normals, preferences, w_sampler, M)
    mean_weight = np.mean(w_samples, axis=0)
    normalized_mean_weight = mean_weight / np.linalg.norm(mean_weight)
    np.save(outdir / "mean_reward.npy", normalized_mean_weight)


def get_simulated_feedback(
    simulation_object: Driver,
    input_A: np.ndarray,
    input_B: np.ndarray,
    reward: np.ndarray,
    fake_noise: bool = False,
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


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_random_inputs(
    simulation_object: Driver, b: int
) -> Tuple[np.ndarray, np.ndarray]:
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

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
    return a_inputs, b_inputs


def load_true_reward(
    true_reward: Tuple[float, float, float, float], outdir: Path
) -> np.ndarray:
    true_reward_path = outdir / "true_reward.npy"
    if true_reward_path.is_file():
        return np.load(true_reward_path)
    else:
        return normalize(np.array(true_reward))


def load(outdir: Path, filename: str, overwrite: bool) -> Optional[np.ndarray]:
    if overwrite:
        return None

    filepath = outdir / filename
    if filepath.exists():
        return np.load(filepath)

    return None


@arg("--true-reward", nargs=4, type=float)
def batch(
    *,
    N: int = 100,
    M: int = 1000,
    b: int = 10,
    outdir: Path = Path("questions"),
    simulate_human: bool = False,
    true_reward: Tuple[float, float, float, float] = random((4,)) * 2 - 1,
    fake_noise: bool = False,
    overwrite: bool = False,
):
    if N % b != 0:
        print("N must be divisible to b")
        exit(0)
    B = 20 * b

    get_prefs: GetPrefFunc
    if simulate_human:
        true_reward = load_true_reward(true_reward, outdir)
        np.save(outdir / "true_reward.npy", true_reward)

        get_prefs = partial(
            get_simulated_feedback, reward=true_reward, fake_noise=fake_noise
        )
    else:
        get_prefs = get_feedback

    simulation_object = create_env("driver")

    w_sampler = Sampler(simulation_object.num_of_features)
    normals: np.ndarray = load(outdir, filename="normals.npy", overwrite=overwrite)
    preferences: np.ndarray = load(
        outdir, filename="preferences.npy", overwrite=overwrite
    )
    inputs: np.ndarray = load(outdir, filename="inputs.npy", overwrite=overwrite)

    if len(inputs) == 0:
        # If this is the first iteration, make random inputs
        a_inputs, b_inputs = make_random_inputs(simulation_object, b)
        inputs = update_inputs(a_inputs, b_inputs, inputs, outdir)
        normals, preferences = elicit_preferences(
            a_inputs,
            b_inputs,
            simulation_object,
            normals,
            preferences,
            outdir,
            get_prefs,
        )
    elif len(inputs) * b > len(preferences):
        # If we exit in the middle of a batch, we need to finish up the rest of the batch
        last_index = len(preferences) % b
        a_inputs, b_inputs = inputs[-1, :, last_index:]
        assert len(a_inputs) == len(inputs) * b - len(preferences)
        normals, preferences = elicit_preferences(
            a_inputs,
            b_inputs,
            simulation_object,
            normals,
            preferences,
            outdir,
            get_prefs,
        )

    try:
        for _ in range(len(inputs) * b, N, b):
            if not simulate_human:
                print("Thinking...")
            a_inputs, b_inputs = sample_inputs(
                normals, preferences, w_sampler, simulation_object, M, b, B
            )
            inputs = update_inputs(a_inputs, b_inputs, inputs, outdir)
            normals, preferences = elicit_preferences(
                a_inputs,
                b_inputs,
                simulation_object,
                normals,
                preferences,
                outdir,
                get_prefs,
            )
    except KeyboardInterrupt:
        # Pass through to finally
        print("Saving results, please do not exit again.")
    finally:
        save_reward(normals, preferences, w_sampler, M, outdir)


if __name__ == "__main__":
    argh.dispatch_command(batch)
