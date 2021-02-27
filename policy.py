import logging
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

import driver  # type: ignore
import fire  # type: ignore
import gym  # type: ignore
import numpy as np
from driver.gym_driver import GymDriver  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from torch.utils.tensorboard import SummaryWriter

from active.simulation_utils import create_env  # type: ignore
from collect_gt_alignment import make_path
from TD3.TD3 import TD3  # type: ignore
from TD3.utils import ReplayBuffer  # type: ignore
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
    actor_layers: List[int] = [256, 256],
    critic_layers: List[int] = [256, 256],
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
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    logging.basicConfig(level=verbosity)

    if n_replications is not None:
        n_replications = int(n_replications)  # Fire is bad about optional arguments

        reward_dir, reward_name = make_reward_path(reward_path)
        Parallel(n_jobs=-2)(
            delayed(train)(
                reward_path=reward_dir / str(i) / reward_name,
                outdir=Path(outdir) / str(i),
                actor_layers=actor_layers,
                critic_layers=critic_layers,
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
    logging.basicConfig(filename=outdir / "log", level=verbosity)

    reward = np.load(reward_path)
    env = gym.make("driver-v1", reward=reward, horizon=horizon, random_start=random_start)
    logging.info("Initialized env")

    state_dim = np.prod(env.observation_space.shape) + reward.shape[0]
    action_dim = np.prod(env.action_space.shape)
    # TODO(joschnei): Clamp expects a float, but we should use the entire vector here.
    max_action = max(np.max(env.action_space.high), -np.min(env.action_space.low))

    td3 = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_kwargs={"layers": actor_layers},
        critic_kwargs={"layers": critic_layers},
        writer=writer,
    )
    logging.info("Initialized TD3 algorithm")

    if (outdir / model_name).exists():
        td3.load(outdir / model_name)
        logging.info("Loaded existing TD3 model weights")

    buffer = ReplayBuffer(state_dim, action_dim, writer=writer)

    raw_state = env.reset()
    state = make_TD3_state(raw_state, reward_features=GymDriver.get_features(raw_state))

    episode_reward_feautures = np.empty((env.horizon, *env.reward.shape))
    episode_actions = np.empty((env.horizon, *env.action_space.shape))
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

        reward_features = info["reward_features"]
        writer.add_scalar("reward", reward, t)
        writer.add_scalar("R/lane", reward_features[0], t)
        writer.add_scalar("R/speed", reward_features[1], t)
        writer.add_scalar("R/heading", reward_features[2], t)
        writer.add_scalar("R/dist", reward_features[3], t)
        writer.add_scalar("A/accel", action[1], t)
        writer.add_scalar("A/turn", action[0], t)

        if save_period - (t % save_period) <= env.horizon:
            episode_reward_feautures[t % env.horizon] = reward_features
            episode_actions[t % env.horizon] = action

        if done:
            assert t % horizon == horizon - 1, f"Done at t={t} when horizon={horizon}"
            raw_state = env.reset()
            state = make_TD3_state(raw_state, reward_features=GymDriver.get_features(raw_state))
        else:
            next_state = make_TD3_state(next_raw_state, reward_features)

        # Store data in replay buffer
        buffer.add(state, action, next_state, reward, done=float(done))

        state = next_state

        # Train agent after collecting sufficient data
        if t >= n_random_timesteps:
            td3.train(buffer, batch_size)

        if t % save_period == 0:
            logging.info(f"{t} / {n_timesteps}")

            if t != 0:
                plot_heading(
                    heading=episode_reward_feautures[:, 2],
                    outdir=outdir,
                    name=str(t // save_period),
                )
                plot_turn(turn=episode_actions[:, 0], outdir=outdir, name=str(t // save_period))
            td3.save(str(outdir / model_name))


def plot_heading(heading: np.ndarray, outdir: Path, name: str = "") -> None:
    plt.plot(heading)
    plt.xlabel("Timestep")
    plt.ylabel("Heading")

    name = "heading.png" if name == "" else f"heading_{name}.png"
    plt.savefig(outdir / name)
    plt.close()


def plot_turn(turn: np.ndarray, outdir: Path, name: str = "") -> None:
    plt.plot(turn)
    plt.xlabel("Timestep")
    plt.ylabel("Turn")

    name = "turn.png" if name == "" else f"turn_{name}.png"
    plt.savefig(outdir / name)
    plt.close()


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
