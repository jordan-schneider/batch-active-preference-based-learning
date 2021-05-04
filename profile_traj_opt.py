import logging
import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple

import fire  # type: ignore
import numpy as np
from driver.gym.legacy_env import LegacyEnv
from tensorflow.keras.optimizers import SGD, Adam  # type: ignore

from active.simulation_utils import TrajOptimizer
from utils import parse_replications


def main(
    mistakes_path: Path,
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
        for index in replication_indices:
            main(
                mistakes_path=mistakes_path / str(index) / "planner_mistakes.pkl",
                plan_iters=plan_iters,
                optim=optim,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                verbosity=verbosity,
            )
        exit()

    mistakes: Tuple[np.ndarray, np.ndarray, np.ndarray] = pickle.load(open(mistakes_path, "rb"))

    for start, reward_weights, actions in mistakes:
        env = LegacyEnv(reward=reward_weights)

        if optim == "sgd":
            optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov)
        elif optim == "adam":
            optimizer = Adam(learning_rate=lr)

        traj_opt = TrajOptimizer(n_planner_iters=plan_iters, optim=optimizer)
        traj = traj_opt.make_opt_traj(reward_weights)

        traj_opt_return = rollout(actions=traj, env=env, start=start)
        policy_return = rollout(actions=actions, env=env, start=start)

        logging.info(
            f"Traj opt return={traj_opt_return}, policy_return={policy_return}, delta={traj_opt_return-policy_return}"
        )


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
