import logging
import pickle
from pathlib import Path
from time import perf_counter
from typing import Literal, Optional, Sequence, Tuple, Union

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from driver.gym_env.legacy_env import LegacyEnv
from tensorflow.keras.optimizers import SGD, Adam  # type: ignore

from active.simulation_utils import TrajOptimizer
from utils import parse_replications, setup_logging


def main(
    mistakes_path: Path,
    outdir: Path,
    plan_iters: int = 10,
    optim: Literal["sgd", "adam"] = "sgd",
    lr: float = 0.1,
    momentum: bool = False,
    nesterov: bool = False,
    extra_inits: bool = False,
    replications: Optional[str] = None,
    log_time: bool = False,
    log_best_inits: bool = False,
    n_traj_max: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    outdir = Path(outdir)
    experiment_dir = outdir / make_experiment(
        optim, lr, plan_iters, momentum, nesterov, extra_inits
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(verbosity=verbosity, log_path=experiment_dir / "log.txt")

    if replications is not None:
        replication_indices = parse_replications(replications)
        mistakes_paths = [
            Path(mistakes_path) / str(index) / "planner_mistakes.pkl"
            for index in replication_indices
        ]
    else:
        mistakes_paths = [Path(mistakes_path)]

    if optim == "sgd":
        optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov)
    elif optim == "adam":
        optimizer = Adam(learning_rate=lr)

    env = LegacyEnv(reward=np.zeros(4))

    starts, rewards, better_trajs = collect_mistakes(
        mistakes_paths=mistakes_paths, n_max=n_traj_max
    )

    init_controls = (
        np.array(
            [
                [[0.0, 1.0]] * 50,
                [[0.0, -1.0]] * 50,
                [[-0.5, -1.0]] * 50,
                [[0.5, -1.0]] * 50,
                [[0.5, 1.0]] * 50,
                [[-0.5, 1.0]] * 50,
            ]
        )
        if extra_inits
        else None
    )

    logging.info("Making trajectories")
    opt_trajs, losses = make_opt_trajs(
        traj_opt=TrajOptimizer(
            n_planner_iters=plan_iters,
            optim=optimizer,
            init_controls=init_controls,
            log_best_init=log_best_inits,
        ),
        rewards=rewards,
        starts=starts,
        log_time=log_time,
    )

    logging.info("Rolling out trajectories")
    returns = np.empty((len(starts), 2))
    for i, (start, reward_weights, opt_traj, policy_traj, loss) in enumerate(
        zip(starts, rewards, opt_trajs, better_trajs, losses)
    ):
        env.reward = reward_weights

        traj_opt_return = rollout(actions=opt_traj, env=env, start=start)
        policy_return = rollout(actions=policy_traj, env=env, start=start)

        assert (
            abs(traj_opt_return + loss) < 0.001
        ), f"Rollout={traj_opt_return} and loss={loss}, differ by too much. start={start}, reward={reward_weights}"

        returns[i, 0] = traj_opt_return
        returns[i, 1] = policy_return

        logging.debug(
            f"Traj opt return={traj_opt_return}, loss={loss}, policy_return={policy_return}, delta={traj_opt_return-policy_return}"
        )

    np.save(experiment_dir / "returns.npy", returns)

    deltas = returns[:, 0] - returns[:, 1]

    logging.info(
        f"Mean delta={np.mean(deltas)}, mean better={np.mean(deltas > 0)*100:.1f}%, optim={optim}, lr={lr}, n={plan_iters}, momentum={momentum}, nesterov={nesterov}, extra inits={extra_inits}"
    )

    plot_returns(returns, experiment_dir)


def make_opt_trajs(
    traj_opt: TrajOptimizer,
    rewards: np.ndarray,
    starts: np.ndarray,
    log_time: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    trajs = []
    losses = []
    times = []
    for reward, start_state in zip(rewards, starts):
        start = perf_counter()
        traj, loss = traj_opt.make_opt_traj(reward, start_state, return_loss=True)
        stop = perf_counter()

        trajs.append(traj)
        losses.append(loss)

        times.append(stop - start)

    trajs_array = np.array(trajs)
    assert len(trajs_array.shape) == 3
    assert trajs_array.shape[1:] == (50, 2)

    if log_time:
        logging.info(f"Mean traj opt time={np.mean(times)}")
    return trajs_array, np.array(losses)


def collect_mistakes(
    mistakes_paths: Sequence[Path], n_max: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        if n_max is not None and len(starts) >= n_max:
            break

    starts_array = np.concatenate(starts)
    rewards_array = np.concatenate(rewards)
    trajs_array = np.concatenate(trajs)

    if n_max is not None and starts_array.shape[0] > n_max:
        starts_array = starts_array[:n_max]
        rewards_array = rewards_array[:n_max]
        trajs_array = trajs_array[:n_max]

    assert len(starts_array.shape) == 3
    assert starts_array.shape[1:] == (
        2,
        4,
    ), f"Start state array has shape {starts_array.shape}, expected (n, 2, 4)"

    return starts_array, rewards_array, trajs_array


def make_experiment(
    optim: str, lr: float, plan_iters: int, momentum: bool, nesterov: bool, extra_inits: bool
) -> str:
    experiment = f"{optim}_{lr}_{plan_iters}"

    if momentum:
        momentum_str = "nesterov" if nesterov else "momentum"
        experiment += f"_{momentum_str}"

    if extra_inits:
        experiment += "_inits"

    return experiment


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
    state = start
    traj_return = 0.0
    for action in actions:
        logging.debug(f"before state={state}")
        state, reward, done, info = env.step(action)
        logging.debug(f"after state={state}, action={action}, reward={reward}")
        traj_return += reward
    return traj_return


if __name__ == "__main__":
    fire.Fire(main)
