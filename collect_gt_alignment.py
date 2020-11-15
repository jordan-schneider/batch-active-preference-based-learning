from functools import partial
from itertools import islice
from pathlib import Path
from typing import List, Optional

import fire  # type: ignore
import numpy as np
from joblib import Parallel, delayed  # type: ignore
from numpy.random import default_rng

from demos import append, load
from simulation_utils import compute_best, create_env


def make_path(reward: np.ndarray) -> np.ndarray:
    simulation_object = create_env("driver")
    optimal_ctrl = compute_best(
        simulation_object=simulation_object, w=reward, iter_count=10
    )
    return optimal_ctrl


def collect(
    outdir: Path,
    n_rewards: int,
    test_reward_path: Optional[Path] = None,
    std: Optional[float] = None,
    mean_reward_path: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    outdir = Path(outdir)

    if test_reward_path is not None:
        rewards = np.load(test_reward_path)
    elif mean_reward_path is not None and n_rewards is not None and std is not None:
        mean_reward = np.load(mean_reward_path)
        rewards = default_rng().normal(
            loc=mean_reward, scale=std, shape=(n_rewards, *mean_reward.shape)
        )
    else:
        raise ValueError("You must supply either a set of test rewards, or a mean reward an std from which to sample test rewards.")

    out_rewards = load(outdir, "test_rewards.npy", overwrite=overwrite)

    paths = load(outdir, "optimal_paths.npy", overwrite=overwrite)
    gt_alignment = load(outdir, "aligment.npy", overwrite=overwrite)

    for reward, optimal_ctrl in Parallel(n_jobs=-2)(
        delayed(make_path)(reward) for reward in islice(rewards, n_rewards)
    ):
        out_rewards = append(out_rewards, reward)
        paths = append(paths, optimal_ctrl)

        simulation_object = create_env("driver")
        simulation_object.set_ctrl(optimal_ctrl)
        simulation_object.watch(1)

        alignment = input("Aligned (y/n):").lower()
        while alignment not in ["y", "n"]:
            alignment = input("Aligned (y/n):").lower()
        gt_alignment = append(gt_alignment, alignment == "y")

    np.save(open(outdir / "test_rewards.npy", "wb"), out_rewards)
    np.save(open(outdir / "optimal_paths.npy", "wb"), np.array(paths))
    np.save(open(outdir / "aligment.npy", "wb"), gt_alignment)


if __name__ == "__main__":
    fire.Fire(collect)
