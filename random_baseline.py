import logging
import pickle
from pathlib import Path
from typing import Literal, Optional

import fire  # type: ignore
import numpy as np
from driver.legacy.models import Driver
from joblib.parallel import Parallel, delayed  # type: ignore
from scipy.linalg import norm  # type: ignore

from active.sampling import Sampler
from active.simulation_utils import get_feedback, get_simulated_feedback
from utils import (
    append,
    load,
    make_reward_path,
    parse_replications,
    safe_normalize,
    save_reward,
    setup_logging,
    update_inputs,
)


def update_response(
    input_features: Optional[np.ndarray],
    normals: Optional[np.ndarray],
    preferences: Optional[np.ndarray],
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
    return input_features, normals, preferences


import multiprocessing


def make_random_questions(n_questions: int, env: Driver) -> np.ndarray:
    inputs = np.random.uniform(
        low=-1,
        high=1,
        size=(n_questions, 2, env.total_time, env.input_size),
    )
    return inputs


def main(
    n_questions: int,
    query_type: Literal["strict", "weak"] = "strict",
    equiv_size: float = 1.1,
    reward_iterations: int = 100,
    outdir: Path = Path("data/simulated/random/elicitation"),
    human: bool = False,
    reward_path: Optional[Path] = None,
    replications: Optional[str] = None,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    setup_logging(verbosity=verbosity, log_path=outpath / "log.txt")

    if not human:
        assert reward_path is not None
        reward_dir, reward_name = make_reward_path(reward_path)
        reward_path = reward_dir / reward_name

    if replications is not None:
        replication_indices = parse_replications(replications)
        n_cpus = min(multiprocessing.cpu_count() - 4, len(replication_indices))
        Parallel(n_jobs=n_cpus)(
            delayed(main)(
                n_questions=n_questions,
                query_type=query_type,
                equiv_size=equiv_size,
                reward_iterations=reward_iterations,
                outdir=outpath / str(i),
                human=human,
                reward_path=reward_dir / str(i) / reward_name,
                overwrite=overwrite,
                verbosity=verbosity,
            )
            for i in replication_indices
        )
        exit()

    if not human:
        assert reward_path is not None
        if not reward_path.exists():
            logging.warning("Reward path given does not exist, generating random reward.")
            true_reward = np.random.default_rng().normal(loc=0, scale=1, size=(4,))
            true_reward = safe_normalize(true_reward)
            np.save(reward_path, true_reward)
        else:
            true_reward = np.load(reward_path)

    pickle.dump(
        {
            "n_questions": n_questions,
            "query_type": query_type,
            "equiv_size": equiv_size,
            "reward_iterations": reward_iterations,
            "human": human,
        },
        open(outpath / "flags.pkl", "wb"),
    )

    normals = load(outpath / "normals.npy", overwrite=overwrite)
    preferences = load(outpath / "preferences.npy", overwrite=overwrite)
    # TODO(joschnei): Make class for inputs, dimensions are too difficult to reason about
    # (N, 2, 100)
    inputs = load(outpath / "inputs.npy", overwrite=overwrite)
    input_features = load(outpath / "input_features.npy", overwrite=overwrite)

    env = Driver()

    if (
        inputs is not None
        and input_features is not None
        and inputs.shape[0] > input_features.shape[0]
    ):
        logging.info("Catching up to previously generated trajectories.")
        input_A, input_B = inputs[-1]

        if human:
            phi_A, phi_B, preference = get_feedback(env, input_A, input_B, query_type)
        else:
            phi_A, phi_B, preference = get_simulated_feedback(
                env, input_A, input_B, query_type, true_reward, equiv_size
            )

        input_features, normals, preferences = update_response(
            input_features, normals, preferences, phi_A, phi_B, preference, outpath
        )

    # Questions and inputs are duplicated, but this keeps everything consistent for the hot-load case
    new_questions = n_questions - inputs.shape[0] if inputs is not None else n_questions
    questions = make_random_questions(n_questions=new_questions, env=env)
    logging.debug(f"questions={questions[:10]}")

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
            phi_A, phi_B, preference = get_feedback(env, input_A, input_B, query_type)
        else:
            phi_A, phi_B, preference = get_simulated_feedback(
                env, input_A, input_B, query_type, true_reward, equiv_size
            )

        input_features, normals, preferences = update_response(
            input_features, normals, preferences, phi_A, phi_B, preference, outpath
        )

    save_reward(
        query_type=query_type,
        true_delta=equiv_size,
        w_sampler=Sampler(env.num_of_features),
        n_reward_samples=reward_iterations,
        outdir=outpath,
    )


if __name__ == "__main__":
    fire.Fire(main)
