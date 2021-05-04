import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import driver.gym  # type: ignore
import fire  # type: ignore
import gym  # type: ignore
import numpy as np
from driver.gym.legacy_env import LegacyEnv
from gym.spaces import Discrete  # type: ignore
from gym.spaces import Tuple as GymTuple
from gym.spaces import flatdim
from joblib import Parallel, delayed  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from TD3 import Td3, load_td3  # type: ignore
from TD3.utils import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from active.simulation_utils import TrajOptimizer  # type: ignore
from utils import make_reward_path, make_td3_paths, make_TD3_state, parse_replications


def make_outdir(outdir: Union[str, Path], timestamp: bool = False) -> Path:
    outdir = Path(outdir)
    if timestamp:
        outdir = outdir / str(datetime.now())

    outdir.mkdir(parents=True, exist_ok=True)

    return outdir


def train(
    outdir: Path,
    reward_path: Path = Path(),
    actor_layers: List[int] = [256, 256],
    critic_layers: List[int] = [256, 256],
    dense: bool = False,
    use_reward_features: bool = True,
    n_timesteps: int = int(1e6),
    n_random_timesteps: int = int(25e3),
    exploration_noise: float = 0.1,
    batch_size: int = 256,
    save_period: int = int(5e3),
    model_name: str = "policy",
    random_rewards: bool = False,
    timestamp: bool = False,
    random_start: bool = False,
    replications: Optional[str] = None,
    plot_episodes: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    logging.basicConfig(level=verbosity)

    if replications is not None:
        replication_indices = parse_replications(replications)

        if not random_rewards:
            reward_dir, reward_name = make_reward_path(reward_path)
            reward_paths = [reward_dir / str(i) / reward_name for i in replication_indices]
        else:
            reward_paths = [None for _ in replication_indices]

        Parallel(n_jobs=-2)(
            delayed(train)(
                reward_path=reward_path,
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
                timestamp=timestamp,
                random_start=random_start,
                random_rewards=random_rewards,
            )
            for i, reward_path in zip(replication_indices, reward_paths)
        )
        exit()

    outdir = make_outdir(outdir, timestamp)

    writer = SummaryWriter(log_dir=outdir)
    logging.basicConfig(filename=outdir / "log", level=verbosity)

    if not random_rewards:
        reward_weights = np.load(reward_path)
        env: LegacyEnv = gym.make(
            "LegacyDriver-v1", reward=reward_weights, random_start=random_start, time_in_state=True
        )
    else:
        env = gym.make("RandomLegacyDriver-v1", random_start=random_start, time_in_state=True)

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
            extra_state_dim=(random_rewards + use_reward_features) * len(env.reward),
        )
    buffer = ReplayBuffer(td3.state_dim + td3.extra_state_dim, td3.action_dim, writer=writer)
    logging.info("Initialized TD3 algorithm")

    raw_state = env.reset()
    logging.debug(f"raw_state={raw_state}")
    state = make_TD3_state(
        raw_state,
        reward_features=env.features(raw_state),
        reward_weights=env.reward_weights if random_rewards else None,
    )

    episode_reward_feautures = np.empty((env.HORIZON, *env.reward.shape))
    episode_actions = np.empty((env.HORIZON, *env.action_space.shape))
    best_return = float("-inf")
    for t in range(n_timesteps):
        action = pick_action(t, n_random_timesteps, env, td3, state, exploration_noise)
        assert action.shape == action_shape, f"Action shape={action.shape}, expected={action_shape}"
        next_raw_state, reward, done, info = env.step(action)
        log_step(next_raw_state, action, reward, info, log_iter=t, writer=writer)

        # Log episode features
        reward_features = info["reward_features"]
        if save_period - (t % save_period) <= env.HORIZON:
            episode_reward_feautures[t % env.HORIZON] = reward_features
            episode_actions[t % env.HORIZON] = action

        if done:
            assert t % env.HORIZON == env.HORIZON - 1, f"Done at t={t} when horizon={env.HORIZON}"
            next_raw_state = env.reset()
            next_state = make_TD3_state(
                raw_state,
                reward_features=env.features(next_raw_state),
                reward_weights=env.reward_weights if random_rewards else None,
            )
        else:
            next_state = make_TD3_state(
                next_raw_state,
                reward_features,
                reward_weights=info["reward_weights"] if random_rewards else None,
            )

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
                    outdir=outdir / "plots" / "heading",
                    name=str(t // save_period),
                )
                plot_turn(
                    turn=episode_actions[:, 0],
                    outdir=outdir / "plots" / "turn",
                    name=str(t // save_period),
                )
            td3.save(str(outdir / model_name))

            logging.info("Evaluating the policy")
            # TODO(joschnei): If random_rewards, generate either a fixed or random-per-eval bag of
            # eval rewards and save the policy if it has better mean return over the bag of
            # eval-rewards. Otherwise just use the fixed reward.
            if random_rewards:
                raise NotImplementedError("Random rewards haven't been fully implemented yet.")
            eval_return = eval(
                reward_weights,
                td3=td3,
                writer=writer,
                log_iter=t // save_period,
            )
            if eval_return > best_return:
                best_return = eval_return
                td3.save(str(outdir / (f"best_{model_name}")))


def pick_action(
    t: int,
    n_random_timesteps: int,
    env: LegacyEnv,
    td3: Td3,
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
    state: LegacyEnv.State,
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
    env: LegacyEnv,
    actor_kwargs: Dict[str, Any] = {},
    critic_kwargs: Dict[str, Any] = {},
    writer: Optional[SummaryWriter] = None,
    reward_condition: bool = False,
    extra_state_dim: int = 0,
) -> Td3:
    """Makes a TD3 object

    Args:
        env (LegacyEnv): Environment the agent is interacting with
        actor_kwargs (Dict[str, Any], optional): Actor keyword arguments. Defaults to {}.
        critic_kwargs (Dict[str, Any], optional): Critic keyword arguments. Defaults to {}.
        writer (Optional[SummaryWriter], optional): tensorboard writer object. Defaults to None.
        reward_condition (bool, optional): If the model should be conditioned on the current reward weights. Defaults to False.

    Returns:
        Td3: [description]
    """
    state_dim = state_dims(env.observation_space)
    logging.debug(f"state_dim={state_dim}")
    action_dim = state_dims(env.action_space)
    # TODO(joschnei): Clamp expects a float, but we should use the entire vector here.
    max_action = max(np.max(env.action_space.high), -np.min(env.action_space.low))

    td3 = Td3(
        state_dim=state_dim,
        extra_state_dim=extra_state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs,
        writer=writer,
    )

    return td3


def state_dims(space: gym.Space) -> int:
    if isinstance(space, Discrete):
        # The whole reason for this function is that time_remaining is both discrete and only takes
        # up one nn input dimension.
        return 1
    elif isinstance(space, GymTuple):
        return sum(state_dims(inner) for inner in space)
    else:
        return flatdim(space)


def eval(
    reward_weights: np.ndarray,
    td3: Td3,
    writer: Optional[SummaryWriter] = None,
    log_iter: Optional[int] = None,
    start_state: Optional[np.ndarray] = None,
    time_in_state: bool = True,
):
    env: LegacyEnv = gym.make("LegacyDriver-v1", reward=reward_weights, time_in_state=time_in_state)

    raw_state = env.reset()
    if start_state is not None:
        env.state = start_state
        raw_state = start_state

    state = make_TD3_state(raw_state, reward_features=env.features(raw_state))
    rewards = []
    for t in range(50):
        raw_state, reward, done, info = env.step(td3.select_action(state))
        state = make_TD3_state(raw_state=raw_state, reward_features=info["reward_features"])
        rewards.append(reward)
    assert done

    empirical_return = np.mean(rewards)
    if writer is not None:
        assert log_iter is not None
        writer.add_scalar("Eval/return", empirical_return, log_iter)

    return empirical_return


def compare(
    reward_path: Path,
    td3_dir: Path,
    outdir: Path,
    planner_iters: int = 10,
    random_start: bool = False,
    n_starts: int = 1,
    replications: Optional[str] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    logging.basicConfig(level=verbosity, format="%(levelname)s:%(asctime)s:%(message)s")
    if replications is not None:
        replication_indices = parse_replications(replications)
        td3_paths = make_td3_paths(Path(td3_dir), replication_indices)
        for replication, td3_path in zip(replication_indices, td3_paths):
            compare(
                reward_path=Path(reward_path) / str(replication) / "true_reward.npy",
                outdir=Path(outdir) / str(replication) / "comparison_plots",
                td3_dir=td3_path,
                planner_iters=planner_iters,
                random_start=random_start,
                n_starts=n_starts,
                verbosity=verbosity,
            )
        exit()

    reward_weights: np.ndarray = np.load(reward_path).astype(np.float32)
    env = gym.make("LegacyDriver-v1", reward=reward_weights, random_start=random_start)
    td3 = load_td3(env, td3_dir)

    traj_optimizer = TrajOptimizer(planner_iters)

    # TODO(joschnei): Also record the trajectory which was better
    class BadPlannerCollection:
        def __init__(self):
            self.states = None
            self.rewards = None

        def append(self, state: np.ndarray, reward: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if self.states is None:
                self.states = np.array([state])
                self.rewards = np.array([reward])
            else:
                self.states = np.append(self.states, state)
                self.rewards = np.append(self.rewards, reward)

            return self.get()

        def get(self) -> Tuple[np.ndarray, np.ndarray]:
            return self.states, self.rewards

    planner_bad = BadPlannerCollection()

    returns = np.empty((n_starts, 2))
    for i in range(n_starts):
        logging.info(f"{i+1}/{n_starts}")
        start_state: np.ndarray = env.reset()

        logging.info("Optimizing traj")
        opt_traj = traj_optimizer.make_opt_traj(reward_weights, start_state)

        logging.info("Executing traj")
        opt_return = 0.0
        for action in opt_traj:
            state, reward, done, info = env.step(action)
            opt_return += reward

        opt_return = opt_return / len(opt_traj)

        logging.info("Evaluating policy")
        empirical_return = eval(
            reward_weights=reward_weights, td3=td3, start_state=start_state, time_in_state=False
        )

        returns[i] = empirical_return, opt_return

        if opt_return < empirical_return:
            planner_bad.append(start_state, reward_weights)

    outdir.mkdir(parents=True, exist_ok=True)

    plt.hist(returns[:, 0], label="Empirical", alpha=0.5)
    plt.hist(returns[:, 1], label="Optimal", alpha=0.5)
    plt.title("Histogram of Optimal vs Empirical returns")
    plt.legend()
    plt.savefig(outdir / "returns.png")
    plt.close()

    regret = returns[:, 1] - returns[:, 0]
    plt.hist(regret)
    plt.title("Histogram of regret")
    plt.savefig(outdir / "regret.png")
    plt.close()
    logging.info(f"Average regret = {np.mean(regret)}, min={np.min(regret)}, max={np.max(regret)}")

    pickle.dump(planner_bad.get(), (outdir / "planner_mistakes.pkl").open("wb"))


def refine(
    reward_path: Path,
    td3_dir: Path,
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
                env_iters=env_iters,
                batch_size=batch_size,
            )
            for i, td3_path in zip(replication_indices, td3_paths)
        )
        exit()
    td3_dir = Path(td3_dir)
    writer = SummaryWriter(log_dir=td3_dir.parent, filename_suffix="refine")

    reward_weights = np.load(reward_path)
    env = gym.make("LegacyDriver-v1", reward=reward_weights)
    td3 = load_td3(env, td3_dir, writer=writer)

    buffer = ReplayBuffer(td3.state_dim, td3.action_dim, writer=writer)
    logging.info("Initialized TD3 algorithm")

    raw_state = env.reset()
    state = make_TD3_state(raw_state, reward_features=env.features(raw_state))

    for t in range(env_iters):
        action = td3.select_action(state)
        next_raw_state, reward, done, info = env.step(action)
        log_step(next_raw_state, action, reward, info, log_iter=t, writer=writer)

        reward_features = info["reward_features"]

        if done:
            assert t % env.HORIZON == env.HORIZON - 1, f"Done at t={t} when horizon={env.HORIZON}"
            next_raw_state = env.reset()
            next_state = make_TD3_state(
                next_raw_state, reward_features=env.main_car.features(next_raw_state, None).numpy()
            )
        else:
            next_state = make_TD3_state(next_raw_state, reward_features)

        # Store data in replay buffer
        buffer.add(state, action, next_state, reward, done=float(done))

        state = next_state

        td3.update_critic(*buffer.sample(batch_size))

        td3.save(str(td3_dir) + "_refined")


def convert_legacy_td3(filename: Union[str, Path], replications: Optional[str] = None) -> None:
    # We used to include reward features as part of state, but now they're extra dimensions. This
    # function converts from the old encoding to the new one.

    if replications is not None:
        replication_indices = parse_replications(replications)
        td3_paths = make_td3_paths(Path(filename), replication_indices)
        for td3_path in td3_paths:
            convert_legacy_td3(td3_path)
        exit()

    # We can't do this if we're adding more dimensions. We just have car state + reward features
    explicit_args = pickle.load(open(str(filename) + ".meta.pkl", "rb"), fix_imports=True)
    assert "extra_state_dim" not in explicit_args.keys()

    env = gym.make("LegacyDriver-v1", reward=np.zeros(4))

    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    explicit_args["extra_state_dim"] = 4
    max_action = max(np.max(env.action_space.high), -np.min(env.action_space.low))
    td3 = Td3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        **explicit_args,
    )
    td3.load(str(filename))

    pickle.dump(explicit_args, open(str(filename) + ".meta.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire()
