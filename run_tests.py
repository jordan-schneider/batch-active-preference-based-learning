""" Runs the test set generated by post.py by generating fake reward weights and seeing how many
are caught by the preferences."""

import logging
import pickle
from itertools import product
from math import log
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Sequence, Set, Tuple, Union, cast

import argh  # type: ignore
import driver
import gym  # type: ignore
import numpy as np
from argh import arg
from driver.gym_driver import GymDriver
from gym.core import Env  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from joblib.parallel import _verbosity_filter  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore

from active.simulation_utils import create_env, make_opt_traj
from policy import make_td3_paths, make_TD3_state
from random_baseline import make_random_questions
from TD3.TD3 import TD3, load_td3  # type: ignore
from testing_factory import TestFactory
from utils import (
    assert_nonempty,
    assert_normals,
    assert_reward,
    assert_rewards,
    get_mean_reward,
    normalize,
    orient_normals,
    parse_replications,
)


def make_gaussian_rewards(
    n_rewards: int,
    use_equiv: bool,
    mean: Optional[np.ndarray] = None,
    cov: Union[np.ndarray, float, None] = None,
    n_reward_features: int = 4,
) -> np.ndarray:
    """ Makes n_rewards uniformly sampled reward vectors of unit length."""
    assert n_rewards > 0
    mean = mean if mean is not None else np.zeros(n_reward_features)
    cov = cov if cov is not None else np.eye(n_reward_features)
    logging.debug(cov)
    dist = multivariate_normal(mean=mean, cov=cov)

    rewards = normalize(dist.rvs(size=n_rewards))
    if use_equiv:
        rewards = np.concatenate((rewards, np.ones((rewards.shape[0], 1))), axis=1)

    assert_rewards(rewards, use_equiv, n_reward_features)

    return rewards


def old_find_reward_boundary(
    normals: np.ndarray, n_rewards: int, reward: np.ndarray, epsilon: float, use_equiv: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Generates n_rewards reward vectors and determines which are aligned. """
    assert_normals(normals, use_equiv)
    assert n_rewards > 0
    assert epsilon >= 0.0
    assert_reward(reward, use_equiv)

    n_reward_features = normals.shape[1]

    cov = 1.0

    rewards = make_gaussian_rewards(n_rewards, use_equiv, mean=reward, cov=cov)
    normals = normals[reward @ normals.T > epsilon]
    ground_truth_alignment = cast(np.ndarray, np.all(rewards @ normals.T > 0, axis=1))
    mean_agree = np.mean(ground_truth_alignment)

    while mean_agree > 0.55 or mean_agree < 0.45:
        if mean_agree > 0.55:
            cov *= 1.1
        else:
            cov /= 1.1
        if not np.isfinite(cov) or cov <= 0.0 or cov >= 100.0:
            # TODO(joschnei): Break is a code smell
            logging.warning(f"cov={cov}, using last good batch of rewards.")
            break
        rewards = make_gaussian_rewards(n_rewards, use_equiv, mean=reward, cov=cov)
        normals = normals[reward @ normals.T > epsilon]
        ground_truth_alignment = cast(np.ndarray, np.all(rewards @ normals.T > 0, axis=1))
        mean_agree = np.mean(ground_truth_alignment)

    assert ground_truth_alignment.shape == (n_rewards,)
    assert rewards.shape == (n_rewards, n_reward_features)

    return rewards, ground_truth_alignment


def find_reward_boundary(
    true_reward: np.ndarray,
    td3_dir: Path,
    n_rewards: int,
    use_equiv: bool,
    epsilon: float,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # TODO(joschnei): Un-hardcode horizon.
    env = gym.make("driver-v1", reward=true_reward, horizon=50)
    td3 = load_td3(env, td3_dir)

    cov = 1.0
    rewards = make_gaussian_rewards(n_rewards, use_equiv, mean=true_reward, cov=cov)
    alignment = rewards_aligned(td3, env, rewards, epsilon, n_samples).cpu().numpy()
    mean_agree = np.mean(alignment)

    while mean_agree > 0.55 or mean_agree < 0.45:
        if mean_agree > 0.55:
            cov *= 1.1
        else:
            cov /= 1.1
        if not np.isfinite(cov) or cov <= 0.0 or cov >= 100.0:
            # TODO(joschnei): Break is a code smell
            logging.warning(f"cov={cov}, using last good batch of rewards.")
            break
        rewards = make_gaussian_rewards(n_rewards, use_equiv, mean=true_reward, cov=cov)
        alignment = rewards_aligned(td3, env, rewards, epsilon, n_samples).cpu().numpy()
        mean_agree = np.mean(alignment)

    assert alignment.shape == (
        n_rewards,
    ), f"Alignment shape={alignment.shape}, expected={(n_rewards,)}"
    assert rewards.shape == (n_rewards, *true_reward.shape)

    return rewards, alignment


def rewards_aligned(
    td3: TD3, env: Env, test_rewards: np.ndarray, epsilon: float, n_samples: int = int(1e4),
):
    state_shape = env.observation_space.sample().shape
    action_shape = env.action_space.sample().shape
    n_rewards = len(test_rewards)

    raw_states = np.array([env.observation_space.sample() for _ in range(n_samples)])
    assert raw_states.shape == (n_samples, *state_shape)
    reward_features = GymDriver.get_feature_batch(raw_states)
    states = make_TD3_state(raw_states, reward_features)
    opt_actions = td3.select_action(states)
    opt_values = td3.critic.Q1(states, opt_actions).reshape(-1)

    reward_feature_shape = reward_features[0].shape

    assert states.shape == (
        n_samples,
        np.prod(state_shape) + np.prod(reward_feature_shape),
    ), f"States shape={states.shape}"
    assert opt_actions.shape == (
        n_samples,
        *action_shape,
    ), f"Actions shape={opt_actions.shape}, expected={(n_samples, *action_shape)}"
    assert opt_values.shape == (n_samples,), f"Value shape={opt_values.shape}"

    actions = np.empty((n_rewards, n_samples, *action_shape))
    for i, reward in enumerate(test_rewards):
        for j, state in enumerate(raw_states):
            actions[i, j] = make_opt_traj(reward, state).reshape(-1, *action_shape)[0]

    values = td3.critic.Q1(states, actions).reshape(-1)

    assert values.shape == (n_rewards, n_samples,), f"Value shape={values.shape}"

    alignment = np.all(opt_values - values < epsilon, axis=1)
    return alignment


def run_test(normals: np.ndarray, test_rewards: np.ndarray, use_equiv: bool) -> np.ndarray:
    """ Returns the predicted alignment of the fake rewards by the normals. """
    assert_normals(normals, use_equiv)
    results = cast(np.ndarray, np.all(np.dot(test_rewards, normals.T) > 0, axis=1))
    return results


def eval_test(
    normals: np.ndarray, rewards: np.ndarray, aligned: np.ndarray, use_equiv: bool
) -> np.ndarray:
    """ Makes a confusion matrix by evaluating a test on the fake rewards. """
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
            y_true=aligned, y_pred=np.ones(aligned.shape, dtype=bool), labels=[False, True],
        )


def make_outname(
    skip_remove_duplicates: bool,
    skip_noise_filtering: bool,
    skip_epsilon_filtering: bool,
    skip_redundancy_filtering: bool,
    base: str = "out",
) -> str:
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


def remove_equiv(preferences: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """ Finds equivalence preferences and removes them + the associated elements of *arrays. """
    indices = preferences != 0
    preferences = preferences[indices]
    out_arrays = list()
    for array in arrays:
        out_arrays.append(array[indices])
    return (preferences, *out_arrays)


def add_equiv_constraints(
    preferences: np.ndarray, normals: np.ndarray, equiv_prob: float
) -> np.ndarray:
    """ Adds equivalence constraints to a set of halspace constraints. """
    out_normals = list()
    for preference, normal in zip(preferences, normals):
        if preference == 0:
            max_return_diff = equiv_prob - log(2 * equiv_prob - 2)
            # w phi >= -max_return_diff
            # w phi + max_reutrn_diff >=0
            # w phi <= max_return diff
            # 0 <= max_return_diff - w phi
            out_normals.append(np.append(normal, [max_return_diff]))
            out_normals.append(np.append(-normals, [max_return_diff]))
        elif preference == 1 or preference == -1:
            out_normals.append(np.append(normal * preference, [0]))

    return np.ndarray(out_normals)


def load(path: Path, overwrite: bool) -> dict:
    if overwrite:
        return dict()
    if path.exists():
        return pickle.load(open(path, "rb"))
    return dict()


Experiment = Tuple[float, float, int]


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


def run_gt_experiment(
    normals: np.ndarray,
    n_rewards: int,
    n_traj_samples: int,
    reward: np.ndarray,
    epsilon: float,
    delta: float,
    use_equiv: bool,
    n_human_samples: int,
    factory: TestFactory,
    input_features: np.ndarray,
    preferences: np.ndarray,
    outdir: Path,
    model_dir: Path,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Tuple[np.ndarray, np.ndarray, Experiment]:
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

    # TODO(joschnei): Move model load out of parallel step to save GPU ram

    # TODO(joschnei): Really need to make this a fixed set common between comparisons.
    rewards, aligned = find_reward_boundary(
        true_reward=reward,
        td3_dir=model_dir,
        n_rewards=n_rewards,
        use_equiv=use_equiv,
        epsilon=epsilon,
        n_samples=n_traj_samples,
    )
    logging.info(f"aligned={np.sum(aligned)}, unaligned={aligned.shape[0] - np.sum(aligned)}")

    filtered_normals = normals[:n_human_samples]
    filtered_normals, indices = factory.filter_halfplanes(
        inputs_features=input_features,
        normals=filtered_normals,
        epsilon=epsilon,
        preferences=preferences,
        delta=delta,
    )

    confusion = eval_test(
        normals=filtered_normals, rewards=rewards, aligned=aligned, use_equiv=use_equiv
    )

    assert confusion.shape == (2, 2)

    return indices, confusion, experiment


def make_experiments(
    epsilons: Sequence[float],
    deltas: Sequence[float],
    n_human_samples: Sequence[int],
    overwrite: bool,
    experiments: Optional[Set[Experiment]] = None,
) -> Generator[Experiment, None, None]:
    if overwrite:
        # TODO(joschnei): This is stupid but I can't be bothered to cast an iterator to a generator.
        for experiment in product(epsilons, deltas, n_human_samples):
            yield experiment
    else:
        for experiment in product(epsilons, deltas, n_human_samples):
            if experiments is None or not (experiment in experiments):
                yield experiment


def make_normals(inputs: np.ndarray, sim, use_equiv: bool):
    assert len(inputs.shape) == 3
    assert inputs.shape[1] == 2
    normals = np.empty(shape=(inputs.shape[0], sim.num_of_features))
    input_features = np.empty(shape=(inputs.shape[0], 2, sim.num_of_features))
    for i, (input_a, input_b) in enumerate(inputs):
        sim.feed(input_a)
        phi_a = np.array(sim.get_features())

        sim.feed(input_b)
        phi_b = np.array(sim.get_features())

        input_features[i] = np.stack((phi_a, phi_b))

        normals[i] = phi_a - phi_b
    assert_normals(normals, use_equiv)
    return input_features, normals


def dedup(
    normals: np.ndarray, preferences: np.ndarray, input_features: np.ndarray, outdir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dedup_normals, indices = TestFactory.remove_duplicates(normals)
    assert np.all(normals[indices] == dedup_normals)
    normals = dedup_normals
    preferences = preferences[indices]
    input_features = input_features[indices]
    logging.info(f"{normals.shape[0]} questions left after deduplicaiton.")
    np.save(outdir / "dedup_normals.npy", normals)
    np.save(outdir / "dedup_preferences.npy", preferences)
    np.save(outdir / "dedup_input_features.npy", input_features)
    return normals, preferences, input_features


def load_elicitation(
    datadir: Path,
    normals_name: Path,
    preferences_name: Path,
    input_features_name: Path,
    n_reward_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = np.load(datadir / normals_name)
    preferences = np.load(datadir / preferences_name)
    input_features = np.load(datadir / input_features_name)

    assert_normals(normals, False, n_reward_features)

    assert_nonempty(normals, preferences, input_features)
    return normals, preferences, input_features


@arg("--epsilons", nargs="+", type=float)
@arg("--deltas", nargs="+", type=float)
@arg("--human-samples", nargs="+", type=int)
def simulated(
    epsilons: List[float] = [0.0],
    deltas: List[float] = [0.05],
    n_rewards: int = 100,
    human_samples: List[int] = [1],
    n_reward_samples: int = 1000,
    n_traj_samples: int = 1000,
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
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    """ Run tests with full data to determine how much reward noise gets """
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

    outdir.mkdir(parents=True, exist_ok=True)

    if n_random_test_questions is not None:
        # Fire defaults to parsing something as a string if its optional
        n_random_test_questions = int(n_random_test_questions)

    flags = pickle.load(open(datadir / flags_name, "rb"))
    query_type = flags["query_type"]
    equiv_probability = flags["equiv_size"]
    sim = create_env(flags["task"])
    n_reward_features = sim.num_of_features

    elicited_normals, elicited_preferences, elicited_input_features = load_elicitation(
        datadir, normals_name, preferences_name, input_features_name, n_reward_features
    )
    true_reward = np.load(datadir / true_reward_name)
    assert_reward(true_reward, False, n_reward_features)

    if not use_equiv:
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

    if use_equiv:
        elicited_normals = add_equiv_constraints(
            elicited_preferences, elicited_normals, equiv_prob=equiv_probability
        )
        true_reward = np.append(true_reward, [1])
    elif query_type == "weak":
        elicited_preferences, elicited_input_features, elicited_normals = remove_equiv(
            elicited_preferences, elicited_input_features, elicited_normals
        )
    assert_normals(elicited_normals, use_equiv)

    if not skip_remove_duplicates:
        elicited_normals, elicited_preferences, elicited_input_features = dedup(
            elicited_normals, elicited_preferences, elicited_input_features, outdir
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
            sim=sim,
            use_equiv=use_equiv,
        )
        assert_normals(normals, use_equiv)
    else:
        preferences = elicited_preferences
        input_features = elicited_input_features
        normals = orient_normals(
            elicited_normals, elicited_preferences, use_equiv, n_reward_features
        )

    for indices, confusion, experiment in Parallel(n_jobs=-2)(
        delayed(run_gt_experiment)(
            normals=normals,
            n_rewards=n_rewards,
            n_traj_samples=n_traj_samples,
            reward=true_reward,
            epsilon=epsilon,
            delta=delta,
            use_equiv=use_equiv,
            n_human_samples=n,
            factory=factory,
            input_features=input_features,
            preferences=preferences,
            outdir=outdir,
            model_dir=model_dir,
            verbosity=verbosity,
        )
        for epsilon, delta, n in experiments
    ):
        minimal_tests[experiment] = indices
        confusions[experiment] = confusion

    pickle.dump(confusions, open(confusion_path, "wb"))
    pickle.dump(minimal_tests, open(test_path, "wb"))


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
    if n_random_test_questions is None:
        raise ValueError(
            "Must supply n_random_test_questions if use_random_test_questions is true."
        )
    mean_reward = get_mean_reward(
        elicited_input_features, elicited_preferences, reward_iterations, query_type, equiv_size,
    )
    inputs = make_random_questions(n_random_test_questions, sim)
    input_features, normals = make_normals(inputs, sim, use_equiv)
    preferences = normals @ mean_reward > 0

    assert preferences.shape == (normals.shape[0],)

    normals = orient_normals(normals, preferences)

    return normals, preferences, input_features


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
    overwrite: bool = False,
):
    outdir.mkdir(parents=True, exist_ok=True)

    flags = pickle.load(open(datadir / flags_name, "rb"))
    query_type = flags["query_type"]
    equiv_probability = flags["equiv_size"]
    sim = create_env(flags["task"])
    n_reward_features = sim.num_of_features

    normals, preferences, input_features = load_elicitation(
        datadir, normals_name, preferences_name, input_features_name, n_reward_features
    )
    assert preferences.shape[0] > 0

    factory = TestFactory(
        query_type=query_type,
        reward_dimension=normals.shape[1],
        equiv_probability=equiv_probability,
        n_reward_samples=n_model_samples,
        use_mean_reward=use_mean_reward,
        skip_noise_filtering=skip_noise_filtering,
        skip_epsilon_filtering=skip_epsilon_filtering,
        skip_redundancy_filtering=skip_redundancy_filtering,
    )

    if use_equiv:
        normals = add_equiv_constraints(preferences, normals, equiv_prob=equiv_probability)
    else:
        if query_type == "weak":
            preferences, input_features, normals = remove_equiv(
                preferences, input_features, normals
            )
        normals = orient_normals(normals, preferences, use_equiv)
    assert_normals(normals, use_equiv)

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

    if rewards_path is None:
        test_rewards = make_gaussian_rewards(n_rewards, use_equiv)
    else:
        test_rewards = np.load(open(rewards_path, "rb"))
    np.save(outdir / "test_rewards.npy", test_rewards)

    experiments = make_experiments(
        epsilons, deltas, human_samples, overwrite, experiments=set(minimal_tests.keys())
    )

    for indices, result, experiment in Parallel(n_jobs=-2)(
        delayed(run_human_experiment)(
            test_rewards,
            normals,
            input_features,
            preferences,
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


if __name__ == "__main__":
    argh.dispatch_commands([simulated, human])
