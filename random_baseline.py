import logging
import pickle
from pathlib import Path
from typing import Optional

import fire  # type: ignore
import numpy as np
from joblib.parallel import Parallel, delayed  # type: ignore
from scipy.linalg import norm  # type: ignore

from active.sampling import Sampler
from active.simulation_utils import create_env, get_feedback, get_simulated_feedback
from elicitation import append, load, save_reward, update_inputs
from utils import make_reward_path


def update_response(
    input_features: Optional[np.ndarray],
    normals: Optional[np.ndarray],
    preferences: Optional[np.ndarray],
    phi_A,
    phi_B,
    preference: int,
    outdir: Path,
):
    input_features = append(input_features, np.stack([phi_A, phi_B]))
    normals = append(normals, phi_A - phi_B)
    preferences = append(preferences, preference)
    np.save(outdir / "input_features.npy", input_features)
    np.save(outdir / "normals.npy", normals)
    np.save(outdir / "preferences.npy", preferences)
    return input_features, normals, preferences


def make_random_questions(n_questions: int, simulation_object) -> np.ndarray:
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]
    inputs = np.random.uniform(
        low=2 * lower_input_bound,
        high=2 * upper_input_bound,
        size=(n_questions, 2 * simulation_object.feed_size),
    )
    inputs = np.reshape(inputs, (n_questions, 2, -1))
    return inputs


def main(
    task: str,
    query_type: str,
    n_questions: int,
    equiv_size: float = 1.1,
    reward_iterations: int = 100,
    outdir: Path = Path("data/simulated/random/elicitation"),
    human: bool = False,
    reward_path: Optional[Path] = None,
    n_replications: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)

    outpath = Path(outdir)

    if not human:
        assert reward_path is not None
        reward_dir, reward_name = make_reward_path(reward_path)
        reward_path = reward_dir / reward_name

    # TODO(joschnei): I could efficiently parallelize this.
    if n_replications is not None:
        Parallel(n_jobs=-2)(
            delayed(main)(
                task=task,
                query_type=query_type,
                n_questions=n_questions,
                equiv_size=equiv_size,
                reward_iterations=reward_iterations,
                outdir=outpath / str(i),
                human=human,
                reward_path=reward_dir / str(i) / reward_name,
                overwrite=overwrite,
            )
            for i in range(1, n_replications + 1)
        )
        exit()

    outpath.mkdir(parents=True, exist_ok=True)

    if not human:
        assert reward_path is not None
        if not reward_path.exists():
            logging.info("Reward path given does not exist, generating random reward.")
            true_reward = np.random.default_rng().normal(loc=0, scale=1, size=(4,))
            true_reward = true_reward / norm(true_reward)
            np.save(reward_path, true_reward)
        else:
            true_reward = np.load(reward_path)

    pickle.dump(
        {
            "task": task,
            "query_type": query_type,
            "n_questions": n_questions,
            "equiv_size": equiv_size,
            "reward_iterations": reward_iterations,
        },
        open(outpath / "flags.pkl", "wb"),
    )

    normals = load(outpath, filename="normals.npy", overwrite=overwrite)
    preferences = load(outpath, filename="preferences.npy", overwrite=overwrite)
    inputs = load(outpath, filename="inputs.npy", overwrite=overwrite)
    input_features = load(outpath, filename="input_features.npy", overwrite=overwrite)

    simulation_object = create_env(task)

    if (
        inputs is not None
        and input_features is not None
        and inputs.shape[0] > input_features.shape[0]
    ):
        logging.info("Catching up to previously generated trajectories.")
        input_A, input_B = inputs[-1]

        if human:
            phi_A, phi_B, preference = get_feedback(simulation_object, input_A, input_B, query_type)
        else:
            phi_A, phi_B, preference = get_simulated_feedback(
                simulation_object, input_A, input_B, query_type, true_reward, equiv_size
            )

        input_features, normals, preferences = update_response(
            input_features, normals, preferences, phi_A, phi_B, preference, outpath
        )

    # Questions and inputs are duplicated, but this keeps everything consistent for the hot-load case
    new_questions = n_questions - inputs.shape[0] if inputs is not None else n_questions
    questions = make_random_questions(
        n_questions=new_questions, simulation_object=simulation_object
    )

    if inputs is not None:
        assert input_features is not None
        assert normals is not None
        assert preferences is not None
        assert inputs.shape[0] == input_features.shape[0]
        assert inputs.shape[0] == normals.shape[0]
        assert inputs.shape[0] == preferences.shape[0]

    for input_A, input_B in questions:
        inputs = update_inputs(input_A, input_B, inputs, outpath)

        if inputs.shape[0] % 10 == 0:
            logging.info(f"{inputs.shape[0]} of {n_questions}")

        if human:
            phi_A, phi_B, preference = get_feedback(simulation_object, input_A, input_B, query_type)
        else:
            phi_A, phi_B, preference = get_simulated_feedback(
                simulation_object, input_A, input_B, query_type, true_reward, equiv_size
            )

        input_features, normals, preferences = update_response(
            input_features, normals, preferences, phi_A, phi_B, preference, outpath
        )

    save_reward(
        query_type=query_type,
        true_delta=equiv_size,
        w_sampler=Sampler(simulation_object.num_of_features),
        n_reward_samples=reward_iterations,
        outdir=outpath,
    )


if __name__ == "__main__":
    fire.Fire(main)
