from pathlib import Path

import fire  # type: ignore
import gym  # type: ignore
import numpy as np

import driver
from TD3.TD3 import TD3
from TD3.utils import ReplayBuffer


def main(
    reward_path: Path,
    outdir: Path,
    n_timesteps: int = int(1e6),
    n_random_timesteps: int = int(25e3),
    exploration_noise: float = 0.1,
    batch_size: int = 256,
    save_period: int = int(5e3),
    horizon: int = 50,
) -> None:
    reward = np.load(reward_path)

    env = gym.make("driver-v1", reward=reward, horizon=horizon)

    state_dim = np.prod(env.observation_space.shape) + reward.shape[0]
    action_dim = np.prod(env.action_space.shape)
    max_action = env.action_space.high

    td3 = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )

    buffer = ReplayBuffer(state_dim, action_dim)

    state = env.reset()

    for t in range(n_timesteps):

        # Select action randomly or according to policy
        if t < n_random_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                td3.select_action(np.array(state))
                + np.random.normal(0, max_action * exploration_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done)

        # Store data in replay buffer
        buffer.add(state, action, next_state, reward, done_bool)

        state = next_state

        # Train agent after collecting sufficient data
        if t >= n_random_timesteps:
            td3.train(buffer, batch_size)

        if t % save_period == 0:
            td3.save(outdir)


if __name__ == "__main__":
    fire.Fire(main)
