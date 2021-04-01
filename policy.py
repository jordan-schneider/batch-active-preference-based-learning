import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import driver  # type: ignore
import fire  # type: ignore
import gym  # type: ignore
import numpy as np
from driver.legacy.gym_driver import GymDriver  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from torch.utils.tensorboard import SummaryWriter

from active.simulation_utils import create_env, make_opt_traj  # type: ignore
from TD3.TD3 import TD3, load_td3  # type: ignore
from TD3.utils import ReplayBuffer  # type: ignore
from utils import make_reward_path, make_td3_paths, make_TD3_state, parse_replications


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
    dense: bool = False,
    n_timesteps: int = int(1e6),
    n_random_timesteps: int = int(25e3),
    exploration_noise: float = 0.1,
    batch_size: int = 256,
    save_period: int = int(5e3),
    model_name: str = "policy",
    horizon: int = 50,
    timestamp: bool = False,
    random_start: bool = False,
    replications: Optional[str] = None,
    plot_episodes: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    logging.basicConfig(level=verbosity)

    if replications is not None:
        replication_indices = parse_replications(replications)

        reward_dir, reward_name = make_reward_path(reward_path)
        Parallel(n_jobs=-2)(
            delayed(train)(
                reward_path=reward_dir / str(i) / reward_name,
                outdir=Path(outdir) / str(i),
                actor_layers=actor_layers,
                critic_layers=critic_layers,
                dense=dense,
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
            for i in replication_indices
        )
        exit()

    outdir = make_outdir(outdir, timestamp)

    writer = SummaryWriter(log_dir=outdir)
    logging.basicConfig(filename=outdir / "log", level=verbosity)

    reward_weights = np.load(reward_path)
    env = gym.make("driver-v1", reward=reward_weights, horizon=horizon, random_start=random_start)
    action_shape = env.action_space.sample().shape
    logging.info("Initialized env")

    if (outdir / (model_name + "_actor")).exists():
        td3 = load_td3(env=env, filename=outdir / model_name, writer=writer)
    else:
        td3 = make_td3(
            env,
            actor_kwargs={"layers": actor_layers, "dense": dense},
            critic_kwargs={"layers": critic_layers, "dense": dense},
            writer=writer,
        )
    buffer = ReplayBuffer(td3.state_dim, td3.action_dim, writer=writer)
    logging.info("Initialized TD3 algorithm")

    raw_state = env.reset()
    state = make_TD3_state(raw_state, reward_features=GymDriver.get_features(raw_state))

    episode_reward_feautures = np.empty((env.horizon, *env.reward_weights.shape))
    episode_actions = np.empty((env.horizon, *env.action_space.shape))
    best_return = float("-inf")
    for t in range(n_timesteps):
        action = pick_action(t, n_random_timesteps, env, td3, state, exploration_noise)
        assert action.shape == action_shape, f"Action shape={action.shape}, expected={action_shape}"
        next_raw_state, reward, done, info = env.step(action)
        log_step(next_raw_state, action, reward, info, log_iter=t, writer=writer)

        # Log episode features
        reward_features = info["reward_features"]
        if save_period - (t % save_period) <= env.horizon:
            episode_reward_feautures[t % env.horizon] = reward_features
            episode_actions[t % env.horizon] = action

        if done:
            assert t % horizon == horizon - 1, f"Done at t={t} when horizon={horizon}"
            next_raw_state = env.reset()
            next_state = make_TD3_state(
                raw_state, reward_features=GymDriver.get_features(next_raw_state)
            )
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

            if plot_episodes and t != 0:
                plot_heading(
                    heading=episode_reward_feautures[:, 2],
                    outdir=outdir,
                    name=str(t // save_period),
                )
                plot_turn(turn=episode_actions[:, 0], outdir=outdir, name=str(t // save_period))
            td3.save(str(outdir / model_name))

            eval_return = eval(
                reward_weights, td3=td3, writer=writer, log_iter=t // save_period, horizon=horizon,
            )
            if eval_return > best_return:
                best_return = eval_return
                td3.save(str(outdir / (f"best_{model_name}")))


def pick_action(
    t: int,
    n_random_timesteps: int,
    env: GymDriver,
    td3: TD3,
    state: np.ndarray,
    exploration_noise: float,
) -> np.ndarray:
    if t < n_random_timesteps:
        return env.action_space.sample()

    return (
        td3.select_action(np.array(state))
        + np.random.normal(0, td3.max_action * exploration_noise, size=td3.action_dim)
    ).clip(-td3.max_action, td3.max_action)


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


def log_step(
    state: np.ndarray,
    action: np.ndarray,
    reward: float,
    info: dict,
    log_iter: int,
    writer: SummaryWriter,
) -> None:
    reward_features = info["reward_features"]
    writer.add_scalar("reward", reward, log_iter)
    writer.add_scalar("R/lane", reward_features[0], log_iter)
    writer.add_scalar("R/speed", reward_features[1], log_iter)
    writer.add_scalar("R/heading", reward_features[2], log_iter)
    writer.add_scalar("R/dist", reward_features[3], log_iter)
    writer.add_scalar("A/accel", action[1], log_iter)
    writer.add_scalar("A/turn", action[0], log_iter)


def make_td3(
    env: GymDriver,
    actor_kwargs: Dict[str, Any] = {},
    critic_kwargs: Dict[str, Any] = {},
    writer: Optional[SummaryWriter] = None,
) -> TD3:
    state_dim = np.prod(env.observation_space.shape) + env.reward_weights.shape[0]
    action_dim = np.prod(env.action_space.shape)
    # TODO(joschnei): Clamp expects a float, but we should use the entire vector here.
    max_action = max(np.max(env.action_space.high), -np.min(env.action_space.low))

    td3 = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs,
        writer=writer,
    )

    return td3


def eval(
    reward_weights: np.ndarray,
    td3: TD3,
    writer: Optional[SummaryWriter] = None,
    log_iter: Optional[int] = None,
    horizon: int = 50,
):
    logging.info("Evaluating the policy")
    env = gym.make("driver-v1", reward=reward_weights, horizon=horizon)

    raw_state = env.reset()
    state = make_TD3_state(raw_state, reward_features=GymDriver.get_features(raw_state))
    rewards = []
    for t in range(horizon):
        raw_state, reward, done, info = env.step(td3.select_action(state))
        state = make_TD3_state(raw_state=raw_state, reward_features=info["reward_features"])
        rewards.append(reward)
    assert done
    empirical_return = np.mean(rewards)
    if writer is not None:
        assert log_iter is not None
        writer.add_scalar("Eval/return", empirical_return, log_iter)
    return empirical_return


def compare(reward_path: Path, td3_dir: Path, horizon: int = 50):
    logging.basicConfig(level="INFO")
    reward_weights = np.load(reward_path)
    env = gym.make("driver-v1", reward=reward_weights, horizon=horizon)
    td3 = load_td3(env, td3_dir)
    empirical_return = eval(reward_weights, td3, horizon=horizon)

    opt_traj = make_opt_traj(reward_weights)
    simulation_object = create_env("driver")
    simulation_object.set_ctrl(opt_traj)
    features = simulation_object.get_features()
    opt_return = reward_weights @ features
    logging.info(f"Optimal return={opt_return}, empirical return={empirical_return}")


def refine(
    reward_path: Path,
    td3_dir: Path,
    horizon: int = 50,
    env_iters: int = int(1e5),
    batch_size: int = 100,
    replications: Optional[Union[str, Tuple[int, ...]]] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    logging.basicConfig(level=verbosity)
    if replications is not None:
        replication_indices = parse_replications(replications)
        reward_dir, reward_name = make_reward_path(reward_path)
        td3_paths = make_td3_paths(Path(td3_dir), replication_indices)
        Parallel(n_jobs=-2)(
            delayed(refine)(
                reward_path=reward_dir / str(i) / reward_name,
                td3_dir=td3_path,
                horizon=horizon,
                env_iters=env_iters,
                batch_size=batch_size,
            )
            for i, td3_path in zip(replication_indices, td3_paths)
        )
        exit()
    td3_dir = Path(td3_dir)
    writer = SummaryWriter(log_dir=td3_dir.parent, filename_suffix="refine")

    reward_weights = np.load(reward_path)
    env = gym.make("driver-v1", reward=reward_weights, horizon=horizon)
    td3 = load_td3(env, td3_dir, writer=writer)

    buffer = ReplayBuffer(td3.state_dim, td3.action_dim, writer=writer)
    logging.info("Initialized TD3 algorithm")

    raw_state = env.reset()
    state = make_TD3_state(raw_state, reward_features=GymDriver.get_features(raw_state))

    for t in range(env_iters):
        action = td3.select_action(state)
        next_raw_state, reward, done, info = env.step(action)
        log_step(next_raw_state, action, reward, info, log_iter=t, writer=writer)

        reward_features = info["reward_features"]

        if done:
            assert t % horizon == horizon - 1, f"Done at t={t} when horizon={horizon}"
            next_raw_state = env.reset()
            next_state = make_TD3_state(
                next_raw_state, reward_features=GymDriver.get_features(next_raw_state)
            )
        else:
            next_state = make_TD3_state(next_raw_state, reward_features)

        # Store data in replay buffer
        buffer.add(state, action, next_state, reward, done=float(done))

        state = next_state

        td3.update_critic(*buffer.sample(batch_size))

        td3.save(str(td3_dir) + "_refined")


if __name__ == "__main__":
    fire.Fire()
