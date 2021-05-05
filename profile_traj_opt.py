import logging
import pickle
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from driver.gym.legacy_env import LegacyEnv
from tensorflow.keras.optimizers import SGD, Adam  # type: ignore

from active.simulation_utils import TrajOptimizer
from utils import parse_replications


def main(
    mistakes_path: Path,
    outdir: Path,
    plan_iters: int = 10,
    optim: Literal["sgd", "adam"] = "sgd",
    lr: float = 0.1,
    momentum: bool = False,
    nesterov: bool = False,
    replications: Optional[str] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    logging.basicConfig(level=verbosity, format="%(levelname)s:%(asctime)s:%(message)s")

    if replications is not None:
        replication_indices = parse_replications(replications)
        mistakes_paths = [
            Path(mistakes_path) / str(index) / "planner_mistakes.pkl"
            for index in replication_indices
        ]
    else:
        mistakes_paths = [Path(mistakes_path)]

    outdir = Path(outdir)
    experiment_dir = outdir / make_experiment(optim, lr, plan_iters, momentum, nesterov)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if optim == "sgd":
        optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov)
    elif optim == "adam":
        optimizer = Adam(learning_rate=lr)

    env = LegacyEnv(reward=np.zeros(4))

    starts, rewards, better_trajs = collect_mistakes(mistakes_paths=mistakes_paths)
    opt_trajs = make_opt_trajs(
        traj_opt=TrajOptimizer(n_planner_iters=plan_iters, optim=optimizer),
        rewards=rewards,
        starts=starts,
    )

    returns = np.empty((len(starts), 2))
    for i, (start, reward_weights, opt_traj, policy_traj) in enumerate(
        zip(starts, rewards, opt_trajs, better_trajs)
    ):
        env.reward = reward_weights

        traj_opt_return = rollout(actions=opt_traj, env=env, start=start)
        policy_return = rollout(actions=policy_traj, env=env, start=start)

        returns[i, 0] = traj_opt_return
        returns[i, 1] = policy_return

        logging.debug(
            f"Traj opt return={traj_opt_return}, policy_return={policy_return}, delta={traj_opt_return-policy_return}"
        )

    np.save(experiment_dir / "returns.npy", returns)

    logging.info(
        f"Mean delta={np.mean(returns[:,0] - returns[:,1])}, optim={optim}, lr={lr}, n={plan_iters}, momentum={momentum}, nesterov={nesterov}"
    )

    plot_returns(returns, experiment_dir)


def make_opt_trajs(traj_opt: TrajOptimizer, rewards: np.ndarray, starts: np.ndarray) -> np.ndarray:
    trajs = [traj_opt.make_opt_traj(reward, start) for reward, start in zip(rewards, starts)]
    trajs_array = np.array(trajs)
    assert len(trajs_array.shape) == 3
    assert trajs_array.shape[1:] == (50, 2)
    return trajs_array


def collect_mistakes(mistakes_paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = []
    rewards = []
    trajs = []
    for path in mistakes_paths:
        logging.debug(f"Loading from {path}")
        new_starts, new_rewards, new_trajs = pickle.load(open(path, "rb"))
        if new_starts is None or new_rewards is None or new_trajs is None:
            logging.warning(f"Skipping empty mistakes file {path}")
            continue
        starts.append(new_starts)
        rewards.append(new_rewards)
        trajs.append(new_trajs)

    starts_array = np.concatenate(starts)
    rewards_array = np.concatenate(rewards)
    trajs_array = np.concatenate(trajs)

    assert len(starts_array.shape) == 3
    assert starts_array.shape[1:] == (
        2,
        4,
    ), f"Start state array has shape {starts_array.shape}, expected (n, 2, 4)"

    return starts_array, rewards_array, trajs_array


def make_experiment(optim: str, lr: float, plan_iters: int, momentum: bool, nesterov: bool) -> str:
    if momentum:
        momentum_str = "nesterov" if nesterov else "momentum"
        return f"{optim}_{lr}_{plan_iters}_{momentum_str}"

    return f"{optim}_{lr}_{plan_iters}"


def plot_returns(returns: np.ndarray, outdir: Path):
    assert len(returns.shape) == 2
    assert returns.shape[1] == 2

    deltas = returns[:, 0] - returns[:, 1]
    plt.hist(deltas)
    plt.xlabel("Traj - policy (return)")
    plt.title("Histogram of return difference traj opt and policy rollouts")

    plt.savefig(outdir / "delta.svg")
    plt.close()


def rollout(actions: np.ndarray, env: LegacyEnv, start: np.ndarray) -> float:
    env.reset()
    env.state = start
    traj_return = 0.0
    for action in actions:
        state, reward, done, info = env.step(action)
        traj_return += reward
    return traj_return


if __name__ == "__main__":
    fire.Fire(main)
