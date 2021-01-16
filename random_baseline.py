import pickle
from pathlib import Path

import fire
import numpy as np

from demos import append, load, save_reward, update_inputs
from sampling import Sampler
from simulation_utils import create_env, get_feedback


def update_response(
    input_features: np.ndarray,
    normals: np.ndarray,
    preferences: np.ndarray,
    phi_A: np.ndarray,
    phi_B: np.ndarray,
    preference: int,
    outdir: Path,
):
    input_features = append(input_features, np.stack([phi_A, phi_B]))
    normals = append(normals, phi_A - phi_B)
    preferences = append(preferences, preference)
    np.save(outdir / "input_features.npy", input_features)
    np.save(outdir / "normals.npy", normals)
    np.save(outdir / "preferences.npy", preferences)


def make_questions(n_questions: int, simulation_object) -> np.ndarray:
    z = simulation_object.feed_size
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]
    return np.random.uniform(low=2 * lower_input_bound, high=2 * upper_input_bound, size=(n_questions, 2 * z))


def main(
    task: str,
    query_type: str,
    n_questions: int,
    delta: float,
    reward_iterations: int,
    outdir: str = "random_questions",
    overwrite: bool = False,
):
    outpath = Path(outdir)

    if not outpath.exists():
        outpath.mkdir()

    pickle.dump(
        {
            "task": task,
            "query_type": query_type,
            "n_questions": n_questions,
            "delta": delta,
            "reward_iterations": reward_iterations,
        },
        open(outpath / "flags.pkl", "wb"),
    )

    normals: np.ndarray = load(outpath, filename="normals.npy", overwrite=overwrite)
    preferences: np.ndarray = load(outpath, filename="preferences.npy", overwrite=overwrite)
    inputs: np.ndarray = load(outpath, filename="inputs.npy", overwrite=overwrite)
    input_features: np.ndarray = load(outpath, filename="input_features.npy", overwrite=overwrite)

    simulation_object = create_env(task)
    # Questions and inputs are duplicated, but this keeps everything consistent for the hot-load case
    questions = make_questions(n_questions=n_questions, simulation_object=simulation_object)

    if inputs.shape[0] > input_features.shape[0]:
        input_A, input_B = inputs[-1]

        phi_A, phi_B, preference = get_feedback(simulation_object, input_A, input_B, query_type)

        update_response(input_features, normals, preferences, phi_A, phi_B, preference, outpath)

    for (input_A, input_B) in questions:
        update_inputs(input_A, input_B, inputs, outpath)

        phi_A, phi_B, preference = get_feedback(simulation_object, input_A, input_B, query_type)

        update_response(input_features, normals, preferences, phi_A, phi_B, preference, outpath)

    save_reward(
        query_type=query_type,
        true_delta=delta,
        w_sampler=Sampler(simulation_object.num_of_features),
        M=reward_iterations,
        outdir=outpath,
    )


if __name__ == "__main__":
    fire.Fire(main)
