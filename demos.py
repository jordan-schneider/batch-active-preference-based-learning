import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np

from sampling import Sampler
from simulation_utils import create_env, get_feedback, run_algo


def load(outdir: Path, filename: str, overwrite: bool) -> Optional[np.ndarray]:
    if overwrite:
        return None

    filepath = outdir / filename
    if filepath.exists():
        return np.load(filepath)

    return None


def save_reward(query_type: str, true_delta: float, w_sampler, M: int, outdir: Path):
    w_samples, _ = w_sampler.sample_given_delta(M, query_type, true_delta)
    mean_weight = np.mean(w_samples, axis=0)
    normalized_mean_weight = mean_weight / np.linalg.norm(mean_weight)
    np.save(outdir / "mean_reward.npy", normalized_mean_weight)


def update_inputs(
    a_inputs: np.ndarray, b_inputs: np.ndarray, inputs: np.ndarray, outdir: Path,
) -> np.ndarray:
    """Adds a new pair of input trajectories (a_inputs, b_inputs) to the inputs list and saves it."""
    inputs = append(inputs, np.stack([a_inputs, b_inputs]))
    np.save(outdir / "inputs.npy", inputs)
    return inputs


def append(a: Optional[np.ndarray], b: Union[np.ndarray, int]) -> np.ndarray:
    if a is None:
        if isinstance(b, np.ndarray):
            return b.reshape(1, *b.shape)
        elif isinstance(b, int):
            return np.array([b])
    else:
        if isinstance(b, np.ndarray):
            return np.append(a, b.reshape((1, *b.shape)), axis=0)
        elif isinstance(b, int):
            return np.append(a, b)


def nonbatch(
    task: str,
    criterion: str,
    query_type: str,
    epsilon: float,
    M: int,
    delta: float,
    outdir: Path = Path("questions"),
    overwrite: bool = False,
):
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    # make this None if you will also learn delta, and change the samplers below
    # from sample_given_delta to sample (and of course remove the true_delta argument)
    pickle.dump(
        {
            "task": task,
            "criterion": criterion,
            "query_type": query_type,
            "epsilon": epsilon,
            "M": M,
            "delta": delta,
        },
        open(outdir / "flags.pkl", "wb"),
    )

    normals: np.ndarray = load(outdir, filename="normals.npy", overwrite=overwrite)
    preferences: np.ndarray = load(
        outdir, filename="preferences.npy", overwrite=overwrite
    )
    inputs: np.ndarray = load(outdir, filename="inputs.npy", overwrite=overwrite)
    input_features: np.ndarray = load(
        outdir, filename="input_features.npy", overwrite=overwrite
    )

    w_sampler = Sampler(d)
    if inputs is not None and preferences is not None:
        for (a_phi, b_phi), preference in zip(input_features, preferences):
            w_sampler.feed(a_phi, b_phi, [preference])
    score = np.inf
    try:
        while score >= epsilon:
            w_samples, delta_samples = w_sampler.sample_given_delta(
                M, query_type, delta
            )

            input_A, input_B, score = run_algo(
                criterion, simulation_object, w_samples, delta_samples
            )

            if score > epsilon:
                update_inputs(
                    a_inputs=input_A, b_inputs=input_B, inputs=inputs, outdir=outdir
                )
                phi_A, phi_B, preference = get_feedback(
                    simulation_object, input_A, input_B, query_type
                )
                input_features = append(input_features, np.stack([phi_A, phi_B]))
                normals = append(normals, phi_A - phi_B)
                preferences = append(preferences, preference)
                np.save(outdir / "input_features.npy", input_features)
                np.save(outdir / "normals.npy", normals)
                np.save(outdir / "preferences.npy", preferences)

                w_sampler.feed(phi_A, phi_B, [preference])
    except KeyboardInterrupt:
        # Pass through to finally
        print("\nSaving results, please do not exit again.")
    finally:
        save_reward(query_type, delta, w_sampler, M, outdir)

