from functools import partial
from pathlib import Path
from typing import List

import fire  # type: ignore
import numpy as np
from joblib import Parallel, delayed  # type: ignore
from numpy.random import default_rng

from demos import append, load
from simulation_utils import compute_best, create_env


def make_paths(mean_reward: np.ndarray, variance: float):
    reward = default_rng().normal(loc=mean_reward, scale=variance)
    reward = reward / np.linalg.norm(reward)

    simulation_object = create_env("driver")
    optimal_ctrl = compute_best(
        simulation_object=simulation_object, w=reward, iter_count=10
    )
    return reward, optimal_ctrl


def collect(
    n_rewards: int,
    variance: float,
    reward_path: Path,
    outdir: Path,
    overwrite: bool = False,
) -> None:
    outdir = Path(outdir)
    mean_reward = np.load(open(reward_path, "rb"))

    rewards = load(outdir, "test_rewards.npy", overwrite=overwrite)
    paths = load(outdir, "optimal_paths.npy", overwrite=overwrite)
    gt_alignment = load(outdir, "aligment.npy", overwrite=overwrite)

    for reward, optimal_ctrl in Parallel(n_jobs=-2)(
        delayed(partial(make_paths, mean_reward=mean_reward, variance=variance))()
        for _ in range(n_rewards)
    ):
        rewards = append(rewards, reward)
        paths = append(paths, optimal_ctrl)

        simulation_object = create_env("driver")
        simulation_object.set_ctrl(optimal_ctrl)
        simulation_object.watch(1)

        alignment = input("Aligned (y/n):").lower()
        while alignment not in ["y", "n"]:
            alignment = input("Aligned (y/n):").lower()
        gt_alignment = append(gt_alignment, alignment == "y")

    np.save(open(outdir / "test_rewards.npy", "wb"), rewards)
    np.save(open(outdir / "optimal_paths.npy", "wb"), np.array(paths))
    np.save(open(outdir / "aligment.npy", "wb"), gt_alignment)


if __name__ == "__main__":
    fire.Fire(collect)
