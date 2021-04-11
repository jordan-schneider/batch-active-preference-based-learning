from itertools import product
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed

from active.simulation_utils import TrajOptimizer


def get_opt_actions(
    rewards: np.ndarray,
    states: np.ndarray,
    optim: TrajOptimizer,
    parallel: Parallel,
    action_shape: Tuple[int, ...] = (2,),
) -> np.ndarray:
    def f(
        rewards: np.ndarray,
        states: np.ndarray,
        optim: TrajOptimizer,
        action_shape: Tuple[int, ...] = (2,),
    ):
        batch_size = rewards.shape[0]
        assert states.shape[0] == batch_size
        opt_actions = np.empty((batch_size, *action_shape))
        for i, (reward, state) in enumerate(zip(rewards, states)):
            path = optim.make_opt_traj(reward, state).reshape(-1, *action_shape)
            opt_actions[i] = path[0]

        return opt_actions

    input_batches = np.array_split(list(product(rewards, states)), parallel.n_jobs)

    return np.concatenate(
        parallel(
            delayed(f)(
                rewards=batch[:, 0],
                states=batch[:, 1],
                optim=optim,
                action_shape=action_shape,
            )
            for batch in input_batches
        )
    ).reshape(len(rewards), len(states), *action_shape)
