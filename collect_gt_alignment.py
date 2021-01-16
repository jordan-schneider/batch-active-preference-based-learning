from functools import partial
from itertools import islice
from pathlib import Path
from typing import List, Optional

import fire  # type: ignore
import numpy as np
from joblib import Parallel, delayed  # type: ignore
from numpy.random import default_rng
from scipy import optimize

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
        raise ValueError(
            "You must either supply a path to the test rewards, or a mean reward and "
            "std from which to sample the test rewards."
        )

    # TODO(joschnei): Handle hot loading in cases where the trajectories were done but the gt
    # alignment wasn't collected, or in cases where new rewards were appended but the trajectories
    # weren't done.

    out_rewards = load(outdir, "test_rewards.npy", overwrite=overwrite)
    out_rewards = append(out_rewards, rewards)
    np.save(open(outdir / "test_rewards.npy", "wb"), out_rewards)

    paths = load(outdir, "optimal_paths.npy", overwrite=overwrite)
    gt_alignment = load(outdir, "aligment.npy", overwrite=overwrite)

    paths = np.array(
        Parallel(n_jobs=-2)(
            delayed(make_path)(reward) for reward in islice(rewards, n_rewards)
        )
    )
    np.save(open(outdir / "optimal_paths.npy", "wb"), np.array(paths))

    simulation_object = create_env("driver")
    for path in paths:
        simulation_object.set_ctrl(path)
        simulation_object.watch(1)

        alignment = input("Aligned (y/n):").lower()
        while alignment not in ["y", "n"]:
            alignment = input("Aligned (y/n):").lower()
        gt_alignment = append(gt_alignment, alignment == "y")

    np.save(open(outdir / "aligment.npy", "wb"), gt_alignment)


if __name__ == "__main__":
    fire.Fire(collect)
