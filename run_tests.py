""" Runs the alignment test generated by elicitation.py on a set of test rewards and reports
performance. """

import logging
import pickle
from functools import partial
from itertools import product
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Sequence, Set, Tuple, Union, cast

import argh  # type: ignore
import driver.gym
import gym  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore

from search import GeometricSearch, TestRewardSearch

tf.config.set_visible_devices([], "GPU")  # Car simulation stuff is faster on cpu
from dataclasses import dataclass

import torch  # type: ignore
from argh import arg
from driver.legacy.gym_driver import GymDriver
from driver.legacy.models import Driver
from gym.core import Env  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from TD3 import Td3, load_td3  # type: ignore

from active.simulation_utils import TrajOptimizer, assert_normals, make_normals, orient_normals
from equiv_utils import add_equiv_constraints, remove_equiv
from policy import make_td3_paths, make_TD3_state
from random_baseline import make_random_questions
from testing_factory import TestFactory
from utils import (
    assert_nonempty,
    assert_reward,
    assert_rewards,
    get_mean_reward,
    load,
    make_gaussian_rewards,
    parse_replications,
)

Experiment = Tuple[float, float, int]


# Top level functions callable from fire


def premake_test_rewards(
    epsilons: List[float] = [0.0],
    n_rewards: int = 100,
    n_test_states: int = 1000,
    true_reward_name: Path = Path("true_reward.npy"),
    flags_name: Path = Path("flags.pkl"),
    datadir: Path = Path(),
    outdir: Path = Path(),
    model_dir: Path = Path(),
    use_equiv: bool = False,
    replications: Optional[Union[str, Tuple[int, ...]]] = None,
    n_cpus: int = 1,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    """ Finds test rewards for each experiment. """
    logging.basicConfig(level=verbosity, format="%(levelname)s:%(asctime)s:%(message)s")
    if replications is not None:
        replication_indices = parse_replications(replications)
        model_dirs = make_td3_paths(model_dir, replication_indices)

        for replication, model_dir in zip(replication_indices, model_dirs):
            if not (datadir / str(replication)).exists():
                logging.warning(f"Replication {replication} does not exist, skipping")
                continue

            premake_test_rewards(
                epsilons=epsilons,
                n_rewards=n_rewards,
                n_test_states=n_test_states,
                true_reward_name=true_reward_name,
                flags_name=flags_name,
                datadir=datadir / str(replication),
                outdir=outdir / str(replication),
                model_dir=model_dir,
                use_equiv=use_equiv,
                n_cpus=n_cpus,
                overwrite=overwrite,
            )
        exit()

    outdir.mkdir(parents=True, exist_ok=True)

    true_reward = np.load(datadir / true_reward_name)
    assert_reward(true_reward, False, 4)

    with Parallel(n_jobs=n_cpus) as parallel:
        make_test_rewards(
            epsilons=epsilons,
            true_reward=true_reward,
            n_rewards=n_rewards,
            n_test_states=n_test_states,
            model_dir=model_dir,
            outdir=outdir,
            parallel=parallel,
            use_equiv=use_equiv,
            overwrite=overwrite,
        )


@arg("--epsilons", nargs="+", type=float)
@arg("--deltas", nargs="+", type=float)
@arg("--human-samples", nargs="+", type=int)
def simulated(
    epsilons: List[float] = [0.0],
    deltas: List[float] = [0.05],
    n_rewards: int = 100,
    human_samples: List[int] = [1],
    n_reward_samples: int = 1000,
    n_test_states: int = 1000,
    input_features_name: Path = Path("input_features.npy"),
    normals_name: Path = Path("normals.npy"),
    preferences_name: Path = Path("preferences.npy"),
    true_reward_name: Path = Path("true_reward.npy"),
    flags_name: Path = Path("flags.pkl"),
    datadir: Path = Path(),
    outdir: Path = Path(),
    model_dir: Path = Path(),
    use_equiv: bool = False,
    use_mean_reward: bool = False,
    use_random_test_questions: bool = False,
    n_random_test_questions: Optional[int] = None,
    skip_remove_duplicates: bool = False,
    skip_noise_filtering: bool = False,
    skip_epsilon_filtering: bool = False,
    skip_redundancy_filtering: bool = False,
    replications: Optional[Union[str, Tuple[int, ...]]] = None,
    n_cpus: int = 1,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    """ Evaluates alignment test generated by ground-truth rewards. """
    logging.basicConfig(level=verbosity)

    if replications is not None:
        replication_indices = parse_replications(replications)
        model_dirs = make_td3_paths(model_dir, replication_indices)

        for replication, model_dir in zip(replication_indices, model_dirs):
            if not (datadir / str(replication)).exists():
                logging.warning(f"Replication {replication} does not exist, skipping")
                continue

            simulated(
                epsilons=epsilons,
                deltas=deltas,
                n_rewards=n_rewards,
                human_samples=human_samples,
                n_reward_samples=n_reward_samples,
                input_features_name=input_features_name,
                normals_name=normals_name,
                preferences_name=preferences_name,
                true_reward_name=true_reward_name,
                flags_name=flags_name,
                datadir=datadir / str(replication),
                outdir=outdir / str(replication),
                model_dir=model_dir,
                use_equiv=use_equiv,
                use_mean_reward=use_mean_reward,
                use_random_test_questions=use_random_test_questions,
                n_random_test_questions=n_random_test_questions,
                skip_remove_duplicates=skip_remove_duplicates,
                skip_noise_filtering=skip_noise_filtering,
                skip_epsilon_filtering=skip_epsilon_filtering,
                skip_redundancy_filtering=skip_redundancy_filtering,
                overwrite=overwrite,
            )
        exit()

    parallel = Parallel(n_jobs=n_cpus)

    outdir.mkdir(parents=True, exist_ok=True)

    if n_random_test_questions is not None:
        # Fire defaults to parsing something as a string if its optional
        n_random_test_questions = int(n_random_test_questions)

    flags = pickle.load(open(datadir / flags_name, "rb"))
    query_type = flags["query_type"]
    equiv_probability = flags["equiv_size"]

    env = Driver()
    n_reward_features = env.num_of_features

    elicited_normals, elicited_preferences, elicited_input_features = load_elicitation(
        datadir=datadir,
        normals_name=normals_name,
        preferences_name=preferences_name,
        input_features_name=input_features_name,
        n_reward_features=n_reward_features,
        use_equiv=use_equiv,
        query_type=query_type,
        equiv_probability=equiv_probability,
        remove_duplicates=not skip_remove_duplicates,
        outdir=outdir,
    )
    true_reward = np.load(datadir / true_reward_name)
    assert_reward(true_reward, False, n_reward_features)

    if use_equiv:
        true_reward = np.append(true_reward, [1])
    else:
        assert not np.any(elicited_preferences == 0)

    factory = TestFactory(
        query_type=query_type,
        reward_dimension=elicited_normals.shape[1],
        equiv_probability=equiv_probability,
        n_reward_samples=n_reward_samples,
        use_mean_reward=use_mean_reward,
        skip_noise_filtering=skip_noise_filtering,
        skip_epsilon_filtering=skip_epsilon_filtering,
        skip_redundancy_filtering=skip_redundancy_filtering,
    )

    confusion_path, test_path = make_outnames(
        outdir,
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
    )
    confusions: Dict[Experiment, np.ndarray] = load(confusion_path, overwrite)
    minimal_tests: Dict[Experiment, np.ndarray] = load(test_path, overwrite)

    experiments = make_experiments(
        epsilons, deltas, human_samples, overwrite, experiments=set(minimal_tests.keys())
    )

    if use_random_test_questions:
        normals, preferences, input_features = make_random_test(
            n_random_test_questions,
            elicited_input_features,
            elicited_preferences,
            reward_iterations=flags["reward_iterations"],
            query_type=query_type,
            equiv_size=flags["equiv_size"],
            sim=env,
            use_equiv=use_equiv,
        )
        assert_normals(normals, use_equiv)
    else:
        preferences = elicited_preferences
        input_features = elicited_input_features
        normals = orient_normals(
            elicited_normals, elicited_preferences, use_equiv, n_reward_features
        )

    test_rewards = make_test_rewards(
        epsilons=epsilons,
        true_reward=true_reward,
        n_rewards=n_rewards,
        n_test_states=n_test_states,
        model_dir=model_dir,
        outdir=outdir,
        parallel=parallel,
        use_equiv=use_equiv,
        overwrite=overwrite,
    )

    for indices, confusion, experiment in parallel(
        delayed(run_gt_experiment)(
            normals=normals,
            test_rewards=test_rewards[epsilon][0],
            test_reward_alignment=test_rewards[epsilon][1],
            epsilon=epsilon,
            delta=delta,
            use_equiv=use_equiv,
            n_human_samples=n,
            factory=factory,
            input_features=input_features,
            preferences=preferences,
            outdir=outdir,
            verbosity=verbosity,
        )
        for epsilon, delta, n in experiments
    ):
        minimal_tests[experiment] = indices
        confusions[experiment] = confusion

    pickle.dump(confusions, open(confusion_path, "wb"))
    pickle.dump(minimal_tests, open(test_path, "wb"))


@arg("--epsilons", nargs="+", type=float)
@arg("--deltas", nargs="+", type=float)
@arg("--human-samples", nargs="+", type=int)
def human(
    epsilons: List[float] = [0.0],
    deltas: List[float] = [0.05],
    n_rewards: int = 10000,
    human_samples: List[int] = [1],
    n_model_samples: int = 1000,
    input_features_name: Path = Path("input_features.npy"),
    normals_name: Path = Path("normals.npy"),
    preferences_name: Path = Path("preferences.npy"),
    flags_name: Path = Path("flags.pkl"),
    datadir: Path = Path("questions"),
    outdir: Path = Path("questions"),
    rewards_path: Optional[Path] = None,
    use_equiv: bool = False,
    use_mean_reward: bool = False,
    skip_remove_duplicates: bool = False,
    skip_noise_filtering: bool = False,
    skip_epsilon_filtering: bool = False,
    skip_redundancy_filtering: bool = False,
    n_cpus: int = 1,
    overwrite: bool = False,
):
    """ Evaluates alignment test elicited from a human. """
    outdir.mkdir(parents=True, exist_ok=True)

    parallel = Parallel(n_jobs=n_cpus)

    flags = pickle.load(open(datadir / flags_name, "rb"))
    query_type = flags["query_type"]
    equiv_probability = flags["equiv_size"]

    sim = Driver()
    n_reward_features = sim.num_of_features

    elicited_normals, elicited_preferences, elicited_input_features = load_elicitation(
        datadir=datadir,
        normals_name=normals_name,
        preferences_name=preferences_name,
        input_features_name=input_features_name,
        n_reward_features=n_reward_features,
        use_equiv=use_equiv,
        query_type=query_type,
        equiv_probability=equiv_probability,
        remove_duplicates=not skip_remove_duplicates,
        outdir=outdir,
    )
    assert elicited_preferences.shape[0] > 0

    factory = TestFactory(
        query_type=query_type,
        reward_dimension=elicited_normals.shape[1],
        equiv_probability=equiv_probability,
        n_reward_samples=n_model_samples,
        use_mean_reward=use_mean_reward,
        skip_noise_filtering=skip_noise_filtering,
        skip_epsilon_filtering=skip_epsilon_filtering,
        skip_redundancy_filtering=skip_redundancy_filtering,
    )

    test_path = outdir / make_outname(
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="indices",
    )
    test_results_path = outdir / make_outname(
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="test_results",
    )

    minimal_tests: Dict[Experiment, np.ndarray] = load(test_path, overwrite)
    results: Dict[Experiment, np.ndarray] = load(test_results_path, overwrite)

    test_rewards = (
        np.load(open(rewards_path, "rb"))
        if rewards_path is not None
        else make_gaussian_rewards(n_rewards, use_equiv)
    )
    np.save(outdir / "test_rewards.npy", test_rewards)

    experiments = make_experiments(
        epsilons, deltas, human_samples, overwrite, experiments=set(minimal_tests.keys())
    )

    for indices, result, experiment in parallel(
        delayed(run_human_experiment)(
            test_rewards,
            elicited_normals,
            elicited_input_features,
            elicited_preferences,
            epsilon,
            delta,
            n,
            factory,
            use_equiv,
        )
        for epsilon, delta, n in experiments
    ):
        minimal_tests[experiment] = indices
        results[experiment] = result

    pickle.dump(minimal_tests, open(test_path, "wb"))
    pickle.dump(results, open(test_results_path, "wb"))


# Test reward generation


def make_test_rewards(
    epsilons: Sequence[float],
    true_reward: np.ndarray,
    n_rewards: int,
    n_test_states: int,
    model_dir: Path,
    outdir: Path,
    parallel: Parallel,
    max_attempts: int = 10,
    use_equiv: bool = False,
    overwrite: bool = False,
):
    """ Makes test rewards sets for every epsilon and saves them to a file. """
    env = gym.make(
        "LegacyDriver-v1",
        reward=np.zeros(
            4,
        ),
    )
    model = load_td3(env, model_dir)
    traj_optimizer = TrajOptimizer(n_planner_iters=10)
    test_rewards: Dict[float, Tuple[np.ndarray, np.ndarray]] = load(
        outdir / "test_rewards.pkl", overwrite=overwrite
    )
    if test_rewards is None:
        test_rewards = {}

    new_epsilons = set(epsilons) - test_rewards.keys()

    test_rewards.update(
        {
            epsilon: find_reward_boundary(
                true_reward=true_reward,
                td3=model,
                traj_optimizer=traj_optimizer,
                n_rewards=n_rewards,
                use_equiv=use_equiv,
                epsilon=epsilon,
                n_samples=n_test_states,
                max_attempts=max_attempts,
                outdir=outdir,
                overwrite=overwrite,
                parallel=parallel,
            )
            for epsilon in new_epsilons
        }
    )
    pickle.dump(test_rewards, open(outdir / "test_rewards.pkl", "wb"))
    del model  # Manually free GPU memory allocation, just in case
    return test_rewards


def find_reward_boundary(
    true_reward: np.ndarray,
    td3: Td3,
    traj_optimizer: TrajOptimizer,
    n_rewards: int,
    use_equiv: bool,
    epsilon: float,
    n_samples: int,
    max_attempts: int,
    outdir: Path,
    parallel: Parallel,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Finds a ballanced set of test rewards according to a critic and epsilon. """
    env = gym.make("LegacyDriver-v1", reward=true_reward)

    new_rewards = partial(
        make_gaussian_rewards, n_rewards=n_rewards, use_equiv=use_equiv, mean=true_reward
    )
    get_alignment = partial(
        rewards_aligned,
        td3=td3,
        traj_optimizer=traj_optimizer,
        env=env,
        epsilon=epsilon,
        parallel=parallel,
        n_test_states=n_samples,
    )

    search = TestRewardSearch.load(epsilon=epsilon, path=outdir / "search.pkl", overwrite=overwrite)
    if search is None:
        search = TestRewardSearch(
            epsilon,
            cov_search=GeometricSearch(start=1.0),
            max_attempts=max_attempts,
            outdir=outdir,
            new_rewards=new_rewards,
            get_alignment=get_alignment,
        )
    else:
        search.new_rewards = new_rewards
        search.get_alignment = get_alignment

    best_test = search.run()

    return best_test.rewards, best_test.alignment


def rewards_aligned(
    td3: Td3,
    traj_optimizer: TrajOptimizer,
    env: Env,
    test_rewards: np.ndarray,
    epsilon: float,
    parallel: Parallel,
    n_test_states: int = int(1e4),
) -> np.ndarray:
    """ Determines the epsilon-alignment of a set of test rewards relative to a critic and epsilon. """
    with torch.no_grad():
        logging.debug("Finding reward alignment")
        state_shape = env.observation_space.sample().shape
        action_shape = env.action_space.sample().shape
        n_rewards = len(test_rewards)

        raw_states = np.array([env.observation_space.sample() for _ in range(n_test_states)])
        assert raw_states.shape == (n_test_states, *state_shape)
        reward_features = GymDriver.get_feature_batch(raw_states)
        states = make_TD3_state(raw_states, reward_features)
        opt_actions = td3.select_action(states)
        opt_values: np.ndarray = (
            td3.critic.Q1(states, opt_actions).reshape(n_test_states).cpu().numpy()
        )

        logging.debug("Opt actions and values found.")

        reward_feature_shape = reward_features[0].shape

        assert states.shape == (
            n_test_states,
            np.prod(state_shape) + np.prod(reward_feature_shape),
        ), f"States shape={states.shape}"
        assert opt_actions.shape == (
            n_test_states,
            *action_shape,
        ), f"Actions shape={opt_actions.shape}, expected={(n_test_states, *action_shape)}"

        actions = get_opt_actions(test_rewards, raw_states, traj_optimizer, parallel, action_shape)

        logging.debug("opt actions found, getting values")

        values: np.ndarray = (
            td3.critic.Q1(np.tile(states, (n_rewards, 1, 1)), actions)
            .reshape(n_rewards, n_test_states)
            .cpu()
            .numpy()
        )

        logging.debug("Values done.")

        alignment = cast(np.ndarray, np.all(opt_values - values < epsilon, axis=1))

        logging.debug("Alignment done")
    return alignment


def get_opt_actions(
    rewards: np.ndarray,
    states: np.ndarray,
    optim: TrajOptimizer,
    parallel: Parallel,
    action_shape: Tuple[int, ...] = (2,),
) -> np.ndarray:

    input_batches = np.array_split(list(product(rewards, states)), parallel.n_jobs)

    logging.debug("Branching")

    return np.concatenate(
        parallel(
            delayed(align_worker)(
                rewards=batch[:, 0],
                states=batch[:, 1],
                optim=optim,
                action_shape=action_shape,
            )
            for batch in input_batches
        )
    ).reshape(len(rewards), len(states), *action_shape)


def align_worker(
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


# Simulated Experiment


def run_gt_experiment(
    normals: np.ndarray,
    test_rewards: np.ndarray,
    test_reward_alignment: np.ndarray,
    epsilon: float,
    delta: float,
    use_equiv: bool,
    n_human_samples: int,
    factory: TestFactory,
    input_features: np.ndarray,
    preferences: np.ndarray,
    outdir: Path,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Tuple[np.ndarray, np.ndarray, Experiment]:
    """ Executes an alignment test on a set of test rewards and records the performance of the test."""
    experiment = (epsilon, delta, n_human_samples)

    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=logdir / f"{epsilon}.{delta}.{n_human_samples}.log",
        filemode="w",
        level=verbosity,
        force=True,
    )
    logging.info(f"Working on epsilon={epsilon}, delta={delta}, n={n_human_samples}")

    # TODO(joschnei): Really need to make this a fixed set common between comparisons.

    filtered_normals = normals[:n_human_samples]
    filtered_normals, indices = factory.filter_halfplanes(
        inputs_features=input_features,
        normals=filtered_normals,
        epsilon=epsilon,
        preferences=preferences,
        delta=delta,
    )

    confusion = eval_test(
        normals=filtered_normals,
        rewards=test_rewards,
        aligned=test_reward_alignment,
        use_equiv=use_equiv,
    )

    assert confusion.shape == (2, 2)

    return indices, confusion, experiment


def eval_test(
    normals: np.ndarray, rewards: np.ndarray, aligned: np.ndarray, use_equiv: bool
) -> np.ndarray:
    """ Evaluates an alignment test on a set of test rewards and reports confusion wrt ground truth. """
    assert rewards.shape[0] == aligned.shape[0]
    assert_rewards(rewards, use_equiv)

    if normals.shape[0] > 0:
        results = run_test(normals, rewards, use_equiv)
        logging.info(
            f"predicted true={np.sum(results)}, predicted false={results.shape[0] - np.sum(results)}"
        )
        return confusion_matrix(y_true=aligned, y_pred=results, labels=[False, True])
    else:
        return confusion_matrix(
            y_true=aligned,
            y_pred=np.ones(aligned.shape, dtype=bool),
            labels=[False, True],
        )


# Human Experiments


def run_human_experiment(
    test_rewards: np.ndarray,
    normals: np.ndarray,
    input_features: np.ndarray,
    preferences: np.ndarray,
    epsilon: float,
    delta: float,
    n_human_samples: int,
    factory: TestFactory,
    use_equiv: bool,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Tuple[np.ndarray, np.ndarray, Experiment]:
    """Distills a set of normals and preferences into a test using the factory, and runs that test on test_rewards

    Args:
        test_rewards (np.ndarray): Rewards to run test on
        normals (np.ndarray): normal vector of halfplane constraints defining test questions
        input_features (np.ndarray): reward features of trajectories in each question
        preferences (np.ndarray): Human provided preference over trajectories
        epsilon (float): Size of minimum value gap required for de-noising
        delta (float): How much of the reward posterior must be over the value gap
        n_human_samples (int): Number of preferences to prune down to
        factory (TestFactory): Factory to produce test questions
        use_equiv (bool): Allow equivalent preference labels?
        verbosity (str): Logging verbosity
    Returns:
        Tuple[np.ndarray, np.ndarray, Experiment]: indices of the selected test questions, test results for each reward, and experimental hyperparameters
    """
    logging.basicConfig(level=verbosity)
    if n_human_samples == -1:
        n_human_samples == normals.shape[0]
    filtered_normals = normals[:n_human_samples]
    filtered_normals, indices = factory.filter_halfplanes(
        inputs_features=input_features,
        normals=filtered_normals,
        epsilon=epsilon,
        preferences=preferences,
        delta=delta,
    )

    experiment = (epsilon, delta, n_human_samples)

    results = run_test(filtered_normals, test_rewards, use_equiv)

    return indices, results, experiment


# Common test utils


def make_experiments(
    epsilons: Sequence[float],
    deltas: Sequence[float],
    n_human_samples: Sequence[int],
    overwrite: bool,
    experiments: Optional[Set[Experiment]] = None,
) -> Generator[Experiment, None, None]:
    """ Yields new experiments (unless overwrite is speificed)"""
    if overwrite:
        # TODO(joschnei): This is stupid but I can't be bothered to cast an iterator to a generator.
        for experiment in product(epsilons, deltas, n_human_samples):
            yield experiment
    else:
        for experiment in product(epsilons, deltas, n_human_samples):
            if experiments is None or not (experiment in experiments):
                yield experiment


def make_random_test(
    n_random_test_questions: Optional[int],
    elicited_input_features: np.ndarray,
    elicited_preferences: np.ndarray,
    reward_iterations: int,
    query_type: str,
    equiv_size: float,
    sim,
    use_equiv: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates an alignment test of randomly generated questions answered according to the mean
    posterior reward.
    """
    if n_random_test_questions is None:
        raise ValueError(
            "Must supply n_random_test_questions if use_random_test_questions is true."
        )
    mean_reward = get_mean_reward(
        elicited_input_features,
        elicited_preferences,
        reward_iterations,
        query_type,
        equiv_size,
    )
    inputs = make_random_questions(n_random_test_questions, sim)
    input_features, normals = make_normals(inputs, sim, use_equiv)
    preferences = normals @ mean_reward > 0

    assert preferences.shape == (normals.shape[0],)

    normals = orient_normals(normals, preferences)

    return normals, preferences, input_features


def run_test(normals: np.ndarray, test_rewards: np.ndarray, use_equiv: bool) -> np.ndarray:
    """ Returns the predicted alignment of the fake rewards by the normals. """
    assert_normals(normals, use_equiv)
    results = cast(np.ndarray, np.all(np.dot(test_rewards, normals.T) > 0, axis=1))
    return results


# IO Utils


def make_outname(
    skip_remove_duplicates: bool,
    skip_noise_filtering: bool,
    skip_epsilon_filtering: bool,
    skip_redundancy_filtering: bool,
    base: str = "out",
) -> str:
    """ Constructs a file name for output files based on flags. """
    outname = base
    if skip_remove_duplicates:
        outname += ".skip_duplicates"
    if skip_noise_filtering:
        outname += ".skip_noise"
    if skip_epsilon_filtering:
        outname += ".skip_epsilon"
    if skip_redundancy_filtering:
        outname += ".skip_lp"
    outname += ".pkl"
    return outname


def make_outnames(
    outdir: Path,
    skip_remove_duplicates: bool,
    skip_noise_filtering: bool,
    skip_epsilon_filtering: bool,
    skip_redundancy_filtering: bool,
) -> Tuple[Path, Path]:
    """ Constructs confusion and index output file names based on flags. """
    confusion_path = outdir / make_outname(
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="confusion",
    )
    test_path = outdir / make_outname(
        skip_remove_duplicates,
        skip_noise_filtering,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="indices",
    )
    return confusion_path, test_path


def load_elicitation(
    datadir: Path,
    normals_name: Union[str, Path],
    preferences_name: Union[str, Path],
    input_features_name: Union[str, Path],
    n_reward_features: int,
    use_equiv: bool,
    query_type: Optional[str] = None,
    equiv_probability: Optional[float] = None,
    remove_duplicates: bool = True,
    outdir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Loads and postprocesses elicitation.py output"""
    normals = np.load(datadir / normals_name)
    preferences = np.load(datadir / preferences_name)
    input_features = np.load(datadir / input_features_name)

    if use_equiv:
        assert equiv_probability is not None
        normals = add_equiv_constraints(preferences, normals, equiv_prob=equiv_probability)
    elif query_type == "weak":
        preferences, normals, input_features = remove_equiv(
            preferences,
            normals,
            input_features,
        )
    if remove_duplicates:
        normals, preferences, input_features = dedup(normals, preferences, input_features, outdir)

    assert_normals(normals, False, n_reward_features)
    assert_nonempty(normals, preferences, input_features)

    return normals, preferences, input_features


def dedup(
    normals: np.ndarray,
    preferences: np.ndarray,
    input_features: np.ndarray,
    outdir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Deduplicates a set of normal vectors and grabs associated preferences and features. """
    dedup_normals, indices = TestFactory.remove_duplicates(normals)
    normals = dedup_normals
    preferences = preferences[indices]
    input_features = input_features[indices]
    logging.info(f"{normals.shape[0]} questions left after deduplicaiton.")
    if outdir is not None:
        np.save(outdir / "dedup_normals.npy", normals)
        np.save(outdir / "dedup_preferences.npy", preferences)
        np.save(outdir / "dedup_input_features.npy", input_features)
    return normals, preferences, input_features


if __name__ == "__main__":
    argh.dispatch_commands([premake_test_rewards, simulated, human])
