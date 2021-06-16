""" Runs the alignment test generated by elicitation.py on a set of test rewards and reports
performance. """

import logging
import pickle as pkl
from functools import partial
from itertools import product
from pathlib import Path
from typing import (
    Dict,
    Generator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import argh  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
from driver.gym_env.legacy_env import LegacyEnv
from gym.spaces import flatten  # type: ignore

from search import GeometricSearch, TestRewardSearch

tf.config.set_visible_devices([], "GPU")  # Car simulation stuff is faster on cpu

from argh import arg
from driver.legacy.models import Driver
from gym.core import Env  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore

from active.simulation_utils import TrajOptimizer, assert_normals, make_normals, orient_normals
from equiv_utils import add_equiv_constraints, remove_equiv
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
    rollout,
    setup_logging,
    shape_compat,
)

Experiment = Tuple[float, Optional[float], int]

input_features_name = Path("input_features.npy")
normals_name = Path("normals.npy")
preferences_name = Path("preferences.npy")
true_reward_name = Path("true_reward.npy")
flags_name = Path("flags.pkl")
use_equiv = False

# Top level functions callable from fire


@arg("--epsilons", nargs="+", type=float)
def premake_test_rewards(
    epsilons: List[float] = [0.0],
    n_rewards: int = 100,
    n_test_states: Optional[int] = None,
    n_gt_test_questions: int = 10000,
    true_reward_name: Path = Path("true_reward.npy"),
    datadir: Path = Path(),
    outdir: Path = Path(),
    replications: Optional[Union[str, Tuple[int, ...]]] = None,
    n_cpus: int = 1,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    """ Finds test rewards for each experiment. """
    outdir.mkdir(parents=True, exist_ok=True)
    # TODO(joschnei): I'm making some dangerous logging decisions. Do I want to append to logs, or
    # give logs unique names? I really need to pick at least one.
    setup_logging(verbosity, log_path=outdir / "log.txt")

    if replications is not None:
        replication_indices = parse_replications(replications)

        for replication in replication_indices:
            if not (datadir / str(replication)).exists():
                logging.warning(f"Replication {replication} does not exist, skipping")
                continue

            premake_test_rewards(
                epsilons=epsilons,
                n_rewards=n_rewards,
                n_test_states=n_test_states,
                n_gt_test_questions=n_gt_test_questions,
                true_reward_name=true_reward_name,
                datadir=datadir / str(replication),
                outdir=outdir / str(replication),
                use_equiv=use_equiv,
                n_cpus=n_cpus,
                overwrite=overwrite,
                verbosity=verbosity,
            )
            logging.info(f"Done with replication {replication}")
        exit()

    true_reward = np.load(datadir / true_reward_name)
    assert_reward(true_reward, False, 4)

    with Parallel(n_jobs=n_cpus) as parallel:
        make_test_rewards(
            epsilons=epsilons,
            true_reward=true_reward,
            n_rewards=n_rewards,
            n_test_states=n_test_states,
            n_gt_test_questions=int(n_gt_test_questions),
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
    n_rewards: int = 100,
    human_samples: List[int] = [1],
    n_reward_samples: int = 1000,
    n_test_states: Optional[int] = None,
    n_gt_test_questions: int = 10000,
    traj_opt: bool = False,
    datadir: Path = Path(),
    outdir: Path = Path(),
    deltas: List[Optional[float]] = [None],
    use_mean_reward: bool = False,
    use_random_test_questions: bool = False,
    n_random_test_questions: Optional[int] = None,
    use_cheating_questions: bool = False,
    skip_remove_duplicates: bool = False,
    skip_epsilon_filtering: bool = False,
    skip_redundancy_filtering: bool = False,
    use_true_epsilon: bool = False,
    legacy_test_rewards: bool = False,
    replications: Optional[Union[str, Tuple[int, ...]]] = None,
    n_cpus: int = 1,
    overwrite_test_rewards: bool = False,
    overwrite_results: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    """ Evaluates alignment test generated by ground-truth rewards. """
    logging.basicConfig(level=verbosity, format="%(levelname)s:%(asctime)s:%(message)s")

    if replications is not None:
        replication_indices = parse_replications(replications)

        for replication in replication_indices:
            if not (datadir / str(replication)).exists():
                logging.warning(f"Replication {replication} does not exist, skipping")
                continue

            logging.info(f"Starting replication {replication}")

            simulated(
                epsilons=epsilons,
                deltas=deltas,
                n_rewards=n_rewards,
                human_samples=human_samples,
                n_reward_samples=n_reward_samples,
                n_test_states=n_test_states,
                n_gt_test_questions=n_gt_test_questions,
                datadir=datadir / str(replication),
                outdir=outdir / str(replication),
                use_mean_reward=use_mean_reward,
                use_random_test_questions=use_random_test_questions,
                use_cheating_questions=use_cheating_questions,
                n_random_test_questions=n_random_test_questions,
                skip_remove_duplicates=skip_remove_duplicates,
                skip_epsilon_filtering=skip_epsilon_filtering,
                skip_redundancy_filtering=skip_redundancy_filtering,
                use_true_epsilon=use_true_epsilon,
                legacy_test_rewards=legacy_test_rewards,
                n_cpus=n_cpus,
                overwrite_test_rewards=overwrite_test_rewards,
                overwrite_results=overwrite_results,
                verbosity=verbosity,
            )
        exit()

    logging.info(f"Using {n_cpus} cpus.")
    parallel = Parallel(n_jobs=n_cpus)

    outdir.mkdir(parents=True, exist_ok=True)

    if n_random_test_questions is not None:
        # Argh defaults to parsing something as a string if its optional
        n_random_test_questions = int(n_random_test_questions)

    flags = pkl.load(open(datadir / flags_name, "rb"))
    query_type = flags["query_type"]
    equiv_probability = flags["equiv_size"]

    env = Driver()
    n_reward_features = env.num_of_features

    logging.info("Loading elicitation results")
    elicited_normals, elicited_preferences, elicited_input_features = load_elicitation(
        datadir=datadir,
        normals_name=normals_name,
        preferences_name=preferences_name,
        input_features_name=input_features_name,
        n_reward_features=n_reward_features,
        use_equiv=use_equiv,
        query_type=query_type,
        equiv_probability=equiv_probability,
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
        skip_dedup=skip_remove_duplicates,
        skip_noise_filtering=True,
        skip_epsilon_filtering=skip_epsilon_filtering,
        skip_redundancy_filtering=skip_redundancy_filtering,
        use_true_epsilon=use_true_epsilon,
        true_reward=true_reward,
    )
    logging.info(
        f"""Filtering settings:
    # reward samples={n_reward_samples},
    use mean reward={use_mean_reward},
    skip duplicates={skip_remove_duplicates}
    skip noise={True}
    skip epsilon={skip_epsilon_filtering}
    skip redundancy={skip_redundancy_filtering}
    use true epsilon={use_true_epsilon}
    """
    )

    confusion_path, test_path = make_outnames(
        outdir,
        skip_remove_duplicates,
        True,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
    )
    confusions: Dict[Experiment, np.ndarray] = load(confusion_path, overwrite_results, default={})
    minimal_tests: Dict[Experiment, np.ndarray] = load(test_path, overwrite_results, default={})

    experiments = make_experiments(
        epsilons, deltas, human_samples, overwrite_results, experiments=set(minimal_tests.keys())
    )

    if use_random_test_questions:
        logging.info("Making random test")
        logging.info(f"True reward: {true_reward}")
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

        good_indices = (true_reward @ normals.T) > 0

        logging.info(f"{np.mean(good_indices)*100:2f}% of new test questions agree with gt reward.")

        if use_cheating_questions:
            logging.info(f"Selecting only questions consistent with gt reward")
            normals = normals[good_indices]
            preferences = preferences[good_indices]
            input_features = input_features[good_indices]

        assert_normals(normals, use_equiv)
    else:
        max_n = max(human_samples)
        preferences = elicited_preferences[:max_n]
        input_features = elicited_input_features[:max_n]
        logging.debug(f"elicited_normals={elicited_normals[:10]}")
        normals = orient_normals(
            elicited_normals[:max_n], preferences, use_equiv, n_reward_features
        )
        logging.debug(f"normals={normals[:10]}")

        assert np.all(true_reward @ normals.T >= 0)

    if not legacy_test_rewards:
        test_rewards = make_test_rewards(
            epsilons=epsilons,
            true_reward=true_reward,
            n_rewards=n_rewards,
            n_test_states=n_test_states,
            n_gt_test_questions=int(n_gt_test_questions),
            traj_opt=traj_opt,
            outdir=outdir,
            parallel=parallel,
            use_equiv=use_equiv,
            overwrite=overwrite_test_rewards,
        )
    else:
        test_rewards = legacy_make_test_rewards(1000, n_rewards, true_reward, epsilons, use_equiv)

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

    pkl.dump(confusions, open(confusion_path, "wb"))
    pkl.dump(minimal_tests, open(test_path, "wb"))


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
    use_mean_reward: bool = False,
    skip_remove_duplicates: bool = False,
    skip_epsilon_filtering: bool = False,
    skip_redundancy_filtering: bool = False,
    n_cpus: int = 1,
    overwrite: bool = False,
):
    """ Evaluates alignment test elicited from a human. """
    outdir.mkdir(parents=True, exist_ok=True)

    parallel = Parallel(n_jobs=n_cpus)

    flags = pkl.load(open(datadir / flags_name, "rb"))
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
    )
    assert elicited_preferences.shape[0] > 0

    factory = TestFactory(
        query_type=query_type,
        reward_dimension=elicited_normals.shape[1],
        equiv_probability=equiv_probability,
        n_reward_samples=n_model_samples,
        use_mean_reward=use_mean_reward,
        skip_dedup=skip_remove_duplicates,
        skip_noise_filtering=True,
        skip_epsilon_filtering=skip_epsilon_filtering,
        skip_redundancy_filtering=skip_redundancy_filtering,
    )

    test_path = outdir / make_outname(
        skip_remove_duplicates,
        True,
        skip_epsilon_filtering,
        skip_redundancy_filtering,
        base="indices",
    )
    test_results_path = outdir / make_outname(
        skip_remove_duplicates,
        True,
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

    pkl.dump(minimal_tests, open(test_path, "wb"))
    pkl.dump(results, open(test_results_path, "wb"))


def compare_test_labels(
    test_rewards_path: Path,
    true_reward_path: Path,
    traj_opt: bool = False,
    elicitation: bool = False,
    replications: Optional[str] = None,
    normals_path: Optional[Path] = None,
):
    if replications is not None:
        raise NotImplementedError("Replications not yet implemented")

    starting_tests: Dict[float, Tuple[np.ndarray, np.ndarray]] = pkl.load(
        open(test_rewards_path, "rb")
    )

    assert not (traj_opt == elicitation), "Provided labels must come from exactly one source"

    class Test(NamedTuple):
        rewards: np.ndarray
        q_labels: np.ndarray
        elicitation_labels: np.ndarray

    test_rewards: Dict[float, Test] = {}
    true_reward = np.load(true_reward_path)
    if traj_opt:
        normals = np.load(normals_path)

        for epsilon, (rewards, q_labels) in starting_tests.items():
            normals = normals[true_reward @ normals.T > epsilon]
            elicitation_labels = run_test(normals, rewards, use_equiv=False)

            test_rewards[epsilon] = Test(
                rewards=rewards, q_labels=q_labels, elicitation_labels=elicitation_labels
            )
    elif elicitation:
        parallel = Parallel(n_cpus=-4)
        env = LegacyEnv(reward=true_reward, random_start=True)
        traj_optimizer = TrajOptimizer(10)
        for epsilon, (rewards, elicitation_labels) in starting_tests.items():
            q_labels = rewards_aligned(
                traj_optimizer=traj_optimizer,
                env=env,
                true_reward=true_reward,
                test_rewards=rewards,
                epsilon=epsilon,
                parallel=parallel,
            )

            test_rewards[epsilon] = Test(
                rewards=rewards, q_labels=q_labels, elicitation_labels=elicitation_labels
            )

    total_agree = 0
    total_rewards = 0
    for epsilon, test in test_rewards.items():
        total_agree += np.sum(test.q_labels == test.elicitation_labels)
        total_rewards += len(test.rewards)

    print(
        f"Critic and superset labels agree on {total_agree / total_rewards * 100 :.1f}% of rewards"
    )


# Test reward generation


def make_test_rewards(
    epsilons: Sequence[float],
    true_reward: np.ndarray,
    n_rewards: int,
    outdir: Path,
    parallel: Parallel,
    n_test_states: Optional[int] = None,
    traj_opt: bool = False,
    max_attempts: int = 10,
    n_gt_test_questions: Optional[int] = None,
    use_equiv: bool = False,
    overwrite: bool = False,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """ Makes test rewards sets for every epsilon and saves them to a file. """
    traj_optimizer = (
        TrajOptimizer(n_planner_iters=100, optim=tf.keras.optimizers.Adam(0.2))
        if traj_opt
        else None
    )

    reward_path = outdir / "test_rewards.pkl"

    test_rewards: Dict[float, Tuple[np.ndarray, np.ndarray]] = load(
        reward_path, overwrite=overwrite
    )
    if test_rewards is None:
        test_rewards = {}
    else:
        logging.info(f"Loading test rewards from {reward_path}")

    new_epsilons = set(epsilons) - test_rewards.keys()

    if len(new_epsilons) > 0:
        logging.info(f"Creating new test rewards for epsilons: {new_epsilons}")

    if (n_test_states is not None and n_test_states > 1) or len(new_epsilons) == 1:
        # Parallelize internally
        test_rewards.update(
            {
                epsilon: find_reward_boundary(
                    true_reward=true_reward,
                    traj_optimizer=traj_optimizer,
                    n_rewards=n_rewards,
                    use_equiv=use_equiv,
                    epsilon=epsilon,
                    n_test_states=n_test_states,
                    max_attempts=max_attempts,
                    outdir=outdir,
                    n_gt_test_questions=n_gt_test_questions,
                    overwrite=overwrite,
                    parallel=parallel,
                )[:2]
                for epsilon in new_epsilons
            }
        )
    else:
        for rewards, alignment, epsilon in parallel(
            delayed(find_reward_boundary)(
                true_reward=true_reward,
                traj_optimizer=traj_optimizer,
                n_rewards=n_rewards,
                use_equiv=use_equiv,
                epsilon=epsilon,
                n_test_states=n_test_states,
                max_attempts=max_attempts,
                n_gt_test_questions=n_gt_test_questions,
                outdir=outdir,
                overwrite=overwrite,
                parallel=None,
            )
            for epsilon in new_epsilons
        ):
            test_rewards[epsilon] = (rewards, alignment)

    logging.info(f"Writing generated test rewards to {reward_path}")
    pkl.dump(test_rewards, open(reward_path, "wb"))
    return test_rewards


def find_reward_boundary(
    true_reward: np.ndarray,
    traj_optimizer: Optional[TrajOptimizer],
    n_rewards: int,
    use_equiv: bool,
    epsilon: float,
    max_attempts: int,
    outdir: Path,
    parallel: Parallel,
    n_test_states: Optional[int] = None,
    n_gt_test_questions: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """ Finds a ballanced set of test rewards according to a critic and epsilon. """
    env = LegacyEnv(reward=true_reward)

    # Don't parallelize here if we're only testing at one state
    logging.debug(f"# test states={n_test_states}")
    parallel = None if n_test_states is None or n_test_states <= 1 else parallel

    new_rewards = partial(
        make_gaussian_rewards, n_rewards=n_rewards, use_equiv=use_equiv, mean=true_reward
    )
    get_alignment = partial(
        rewards_aligned,
        traj_optimizer=traj_optimizer,
        env=env,
        true_reward=true_reward,
        epsilon=epsilon,
        parallel=parallel,
        n_test_states=n_test_states,
        n_questions=n_gt_test_questions,
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

    return best_test.rewards, best_test.alignment, epsilon


def rewards_aligned(
    traj_optimizer: Optional[TrajOptimizer],
    env: Env,
    true_reward: np.ndarray,
    test_rewards: np.ndarray,
    epsilon: float,
    parallel: Optional[Parallel] = None,
    n_test_states: Optional[int] = None,
    n_questions: int = 100000,
    use_equiv: bool = False,
) -> np.ndarray:
    """ Determines the epsilon-alignment of a set of test rewards relative to a critic and epsilon. """
    # This test can produce both false positives and false negatives

    # This test is prone to false positives, but a negative is always a true negative
    gt_test = make_gt_test_align(test_rewards, n_questions, true_reward, epsilon, use_equiv)

    if traj_optimizer is not None:
        traj_opt_alignment = make_traj_opt_align(
            traj_optimizer, env, true_reward, test_rewards, epsilon, parallel, n_test_states
        )
        # Start with traj opt alignment, then mask out all of the rewards that failed the gt test
        # x y z
        # 0 0 0
        # 0 1 0 don't trust y when it says something is aligned if you failed the traj opt
        # 1 0 0 if y says it's misaligned, then it is
        # 1 1 1
        # This is just the & function
        alignment = traj_opt_alignment & gt_test

        n_masked = np.sum(gt_test & np.logical_not(gt_test))
        logging.info(
            f"Trajectory optimization labelling produced at least {n_masked} false positives"
        )
    else:
        alignment = gt_test

    return alignment


def make_gt_test_align(
    test_rewards: np.ndarray,
    n_questions: int,
    true_reward: np.ndarray,
    epsilon: float,
    use_equiv: bool = False,
) -> np.ndarray:
    env = Driver()
    trajs = make_random_questions(n_questions, env)
    _, normals = make_normals(trajs, env, use_equiv)

    value_diff = true_reward @ normals.T
    eps_questions = np.abs(value_diff) > epsilon
    normals = normals[eps_questions]

    gt_pref = value_diff[eps_questions] > 0
    normals = orient_normals(normals, gt_pref, use_equiv)

    alignment = cast(np.ndarray, np.all(test_rewards @ normals.T > 0, axis=1))
    assert alignment.shape == (
        test_rewards.shape[0],
    ), f"alignment shape={alignment.shape} is not expected {test_rewards.shape[0]}"
    return alignment


def make_traj_opt_align(
    traj_optimizer: TrajOptimizer,
    env: Env,
    true_reward: np.ndarray,
    test_rewards: np.ndarray,
    epsilon: float,
    parallel: Optional[Parallel] = None,
    n_test_states: Optional[int] = None,
) -> np.ndarray:
    state_shape = env.observation_space.sample().shape
    action_shape = env.action_space.sample().shape

    if n_test_states is not None:
        raw_states = np.array(
            [
                flatten(env.observation_space, env.observation_space.sample())
                for _ in range(n_test_states)
            ]
        )
    else:
        n_test_states = 1
        raw_states = np.array([env.state])
    assert raw_states.shape == (n_test_states, *state_shape)

    opt_plans = make_plans(
        true_reward.reshape(1, 4),
        raw_states,
        traj_optimizer,
        parallel,
        action_shape,
        memorize=True,
    )
    assert opt_plans.shape == (
        1,
        n_test_states,
        50,
        *action_shape,
    ), f"opt_plans shape={opt_plans.shape} is not expected {(1,n_test_states,50,*action_shape)}"
    opt_values: np.ndarray = rollout_plans(env, opt_plans, raw_states)

    plans = make_plans(test_rewards, raw_states, traj_optimizer, parallel, action_shape)
    assert plans.shape == (
        len(test_rewards),
        n_test_states,
        50,
        *action_shape,
    ), f"plans shape={plans.shape} is not expected {(len(test_rewards),n_test_states,50,*action_shape)}"
    values = rollout_plans(env, plans, raw_states)
    assert values.shape == (
        len(test_rewards),
        n_test_states,
    ), f"Values shape={values.shape} is not expected {(len(test_rewards), n_test_states)}"

    alignment = cast(np.ndarray, np.all(opt_values - values < epsilon, axis=1))
    return alignment


def rollout_plans(env: LegacyEnv, plans: np.ndarray, states: np.ndarray):
    returns = np.empty((plans.shape[0], plans.shape[1]))
    assert len(returns.shape) == 2

    assert len(plans.shape) == 4
    for i in range(plans.shape[0]):
        for j in range(plans.shape[1]):
            returns[i, j] = rollout(plans[i, j], env, states[j])
    return returns


def legacy_make_test_rewards(
    n_questions: int,
    n_rewards: int,
    true_reward: np.ndarray,
    epsilons: List[float],
    use_equiv: bool,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """ Generates n_rewards reward vectors and determines which are aligned. """
    assert n_rewards > 0
    assert_reward(true_reward, use_equiv)

    trajs = make_random_questions(n_questions, Driver())
    _, normals = make_normals(trajs, Driver(), use_equiv)
    gt_pref = true_reward @ normals.T > 0
    normals = orient_normals(normals, gt_pref, use_equiv)
    assert_normals(normals, use_equiv)

    n_reward_features = normals.shape[1]

    test_rewards: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    for epsilon in epsilons:
        assert epsilon >= 0.0

        cov = 1.0

        rewards = make_gaussian_rewards(n_rewards, use_equiv, mean=true_reward, cov=cov)
        normals = normals[true_reward @ normals.T > epsilon]
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
            rewards = make_gaussian_rewards(n_rewards, use_equiv, mean=true_reward, cov=cov)
            normals = normals[true_reward @ normals.T > epsilon]
            ground_truth_alignment = cast(np.ndarray, np.all(rewards @ normals.T > 0, axis=1))
            mean_agree = np.mean(ground_truth_alignment)

        assert ground_truth_alignment.shape == (n_rewards,)
        assert rewards.shape == (n_rewards, n_reward_features)

        test_rewards[epsilon] = (rewards, ground_truth_alignment)

    return test_rewards


def make_plans(
    rewards: np.ndarray,
    states: np.ndarray,
    optim: TrajOptimizer,
    parallel: Optional[Parallel] = None,
    action_shape: Tuple[int, ...] = (2,),
    memorize: bool = False,
) -> np.ndarray:

    assert shape_compat(
        rewards, (-1, 4)
    ), f"rewards shape={rewards.shape} is wrong, expected (-1, 4)"

    if parallel is not None:
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
        ).reshape(len(rewards), len(states), 50, *action_shape)
    else:
        plans = np.empty((len(rewards), len(states), 50, *action_shape))
        for i, reward in enumerate(rewards):
            assert reward.shape == (4,)
            for j, state in enumerate(states):
                traj, _ = optim.make_opt_traj(reward, state, memorize=memorize)
                plans[i, j] = traj.reshape(-1, *action_shape)
        return plans


def align_worker(
    rewards: np.ndarray,
    states: np.ndarray,
    optim: TrajOptimizer,
    action_shape: Tuple[int, ...] = (2,),
):
    batch_size = rewards.shape[0]
    assert states.shape[0] == batch_size
    plans = np.empty((batch_size, 50, *action_shape))
    for i, (reward, state) in enumerate(zip(rewards, states)):
        traj, _ = optim.make_opt_traj(reward, state)
        plans[i] = traj.reshape(-1, *action_shape)

    return plans


# Simulated Experiment


def run_gt_experiment(
    normals: np.ndarray,
    test_rewards: np.ndarray,
    test_reward_alignment: np.ndarray,
    epsilon: float,
    delta: Optional[float],
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
        format="%(levelname)s:%(asctime)s:%(message)s",
    )
    logging.info(f"Working on epsilon={epsilon}, delta={delta}, n={n_human_samples}")

    # TODO(joschnei): Really need to make this a fixed set common between comparisons.

    filtered_normals = normals[:n_human_samples]
    input_features = input_features[:n_human_samples]
    preferences = preferences[:n_human_samples]
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
    logging.basicConfig(level=verbosity, format="%(levelname)s:%(asctime)s:%(message)s")
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
    deltas: Sequence[Optional[float]],
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
    logging.info(f"Mean posterior reward for use in random test: {mean_reward}")
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

    assert_normals(normals, False, n_reward_features)
    assert_nonempty(normals, preferences, input_features)

    return normals, preferences, input_features


if __name__ == "__main__":
    argh.dispatch_commands([premake_test_rewards, simulated, human, compare_test_labels])
