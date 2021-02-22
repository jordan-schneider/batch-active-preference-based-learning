import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import fire  # type: ignore
import gym  # type: ignore
import numpy as np
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter

import driver
from active.simulation_utils import create_env
from collect_gt_alignment import make_path
from driver.gym_driver import GymDriver
from TD3.TD3 import TD3
from TD3.utils import ReplayBuffer
from utils import make_reward_path


def make_TD3_state(raw_state: np.ndarray, reward_features: np.ndarray) -> np.ndarray:
    state = np.concatenate((raw_state.flatten(), reward_features))
    return state


def make_outdir(outdir: Union[str, Path], timestamp: bool = False) -> Path:
    outdir = Path(outdir)
    if timestamp:
        outdir = outdir / str(datetime.now())

    outdir.mkdir(parents=True, exist_ok=True)

    return outdir


def train(
    reward_path: Path,
    outdir: Path,
    n_timesteps: int = int(1e6),
    n_random_timesteps: int = int(25e3),
    exploration_noise: float = 0.1,
    batch_size: int = 256,
    save_period: int = int(5e3),
    model_name: str = "policy",
    horizon: int = 50,
    timestamp: bool = False,
    random_start: bool = False,
    n_replications: Optional[int] = None,
) -> None:
    logging.basicConfig(level="INFO")

    if n_replications is not None:
        n_replications = int(n_replications)  # Fire is bad about optional arguments

        reward_dir, reward_name = make_reward_path(reward_path)
        Parallel(n_jobs=-2)(
            delayed(train)(
                reward_path=reward_dir / str(i) / reward_name,
                outdir=Path(outdir) / str(i),
                n_timesteps=n_timesteps,
                n_random_timesteps=n_random_timesteps,
                exploration_noise=exploration_noise,
                batch_size=batch_size,
                save_period=save_period,
                model_name=model_name,
                horizon=horizon,
                timestamp=timestamp,
                random_start=random_start,
            )
            for i in range(1, n_replications + 1)
        )
        exit()

    outdir = make_outdir(outdir, timestamp)

    writer = SummaryWriter(log_dir=outdir)

    reward = np.load(reward_path)
    env = gym.make("driver-v1", reward=reward, horizon=horizon, random_start=random_start)
    logging.info("Initialized env")

    state_dim = np.prod(env.observation_space.shape) + reward.shape[0]
    action_dim = np.prod(env.action_space.shape)
    # TODO(joschnei): Clamp expects a float, but we should use the entire vector here.
    max_action = max(np.max(env.action_space.high), -np.min(env.action_space.low))

    td3 = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action, writer=writer)
    logging.info("Initialized TD3 algorithm")

    if (outdir / model_name).exists():
        td3.load(outdir / model_name)
        logging.info("Loaded existing TD3 model weights")

    buffer = ReplayBuffer(state_dim, action_dim, writer=writer)

    raw_state = env.reset()
    state = make_TD3_state(raw_state, reward_features=GymDriver.get_features(raw_state))

    for t in range(n_timesteps):
        if t < n_random_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                td3.select_action(np.array(state))
                + np.random.normal(0, max_action * exploration_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_raw_state, reward, done, info = env.step(action)
        writer.add_scalar("reward", reward, t)
        next_state = make_TD3_state(next_raw_state, info["reward_features"])

        # Store data in replay buffer
        buffer.add(state, action, next_state, reward, done=float(done))

        state = next_state

        # Train agent after collecting sufficient data
        if t >= n_random_timesteps:
            td3.train(buffer, batch_size)

        if t % save_period == 0:
            logging.info(f"{t} / {n_timesteps}")
            td3.save(str(outdir / model_name))


def eval(reward_path: Path, td3_dir: Path, horizon: int = 50):
    logging.basicConfig(level="INFO")
    logging.info("Evaluating the policy")
    reward_weights = np.load(reward_path)
    env = gym.make("driver-v1", reward=reward_weights, horizon=horizon)

    state_dim = np.prod(env.observation_space.shape) + reward_weights.shape[0]
    action_dim = np.prod(env.action_space.shape)
    # TODO(joschnei): Clamp expects a float, but we should use the entire vector here.
    max_action = max(np.max(env.action_space.high), -np.min(env.action_space.low))

    td3 = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    td3.load(td3_dir)

    logging.info("Loaded the actor-critic models.")

    raw_state = env.reset()
    state = make_TD3_state(raw_state, reward_features=GymDriver.get_features(raw_state))
    rewards = np.empty((horizon,))
    for t in range(horizon):
        raw_state, reward, _, info = env.step(td3.select_action(state))
        state = make_TD3_state(raw_state=raw_state, reward_features=info["reward_features"])
        rewards[t] = reward
    empirical_return = np.mean(rewards)

    logging.info("Rollout done. Finding optimal path.")

    opt_traj = make_path(reward_weights)
    simulation_object = create_env("driver")
    simulation_object.set_ctrl(opt_traj)
    features = simulation_object.get_features()
    opt_return = reward_weights @ features

    logging.info(f"Optimal return={opt_return}, empirical return={empirical_return}")


if __name__ == "__main__":
    fire.Fire()
