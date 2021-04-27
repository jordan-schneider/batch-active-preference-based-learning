import logging
import pickle as pkl
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import arrow
import fire  # type: ignore
import numpy as np
from numpy.linalg import norm
from scipy.stats import multivariate_normal  # type: ignore

from active.sampling import Sampler


def make_gaussian_rewards(
    n_rewards: int,
    use_equiv: bool,
    mean: Optional[np.ndarray] = None,
    cov: Union[np.ndarray, float, None] = None,
    shape: int = 4,
) -> np.ndarray:
    """ Makes n_rewards uniformly sampled reward vectors of unit length."""
    assert n_rewards > 0
    mean = mean if mean is not None else np.zeros(shape)
    cov = cov if cov is not None else np.eye(shape)
    logging.debug(f"Gaussian covariance={cov}")
    dist = multivariate_normal(mean=mean, cov=cov)

    rewards = normalize(dist.rvs(size=n_rewards))
    if use_equiv:
        rewards = np.concatenate((rewards, np.ones((rewards.shape[0], 1))), axis=1)

    assert_rewards(rewards, use_equiv, shape)

    return rewards


def make_TD3_state(
    raw_state: Union[np.ndarray, Tuple[np.ndarray, int]],
    reward_features: np.ndarray,
    reward_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    time_remaining = None
    if isinstance(raw_state, tuple):
        raw_state, time_remaining = raw_state

    if len(raw_state.shape) == 3:
        # If the length is 3, then I'm getting a batch of unflattened car states without time remaining
        assert raw_state.shape[1:] == (2, 4)

        # Reward features should be a batch of (n, 4) where n is the batch size
        assert len(reward_features.shape) == 2
        assert reward_features.shape[1:] == (
            4,
        ), f"reward features shape={reward_features.shape} when state shape={raw_state.shape}"
        assert raw_state.shape[0] == reward_features.shape[0]

        n = raw_state.shape[0]
        raw_state = raw_state.reshape(n, 8)

        # Return a batch of (n, 8 + 4 = 12) raw input vectors
        state = np.concatenate((raw_state, reward_features), axis=1)
    elif raw_state.shape == (2, 4):
        # We've recieved a single unflattened state, possibly with time remaining.
        assert reward_features.shape == (4,)

        # logging.debug(f"time_remaining={time_remaining}")

        # Return a single flattened (8 + 1? + 4) length vector
        state = np.concatenate(
            (
                raw_state.flatten(),
                [time_remaining] if time_remaining is not None else [],
                reward_features,
            )
        )
    elif len(raw_state.shape) == 2 and raw_state.shape[1] == 9:
        assert raw_state.shape[0] == reward_features.shape[0]
        assert reward_features.shape[1] == 4
        # Return (n, 9 + 4) full batch
        state = np.concatenate(raw_state, reward_features, axis=1)
    else:
        raise ValueError(f"{raw_state} is not a valid state or state batch")

    if reward_weights is not None:
        logging.debug("Appending reward weights")
        axis = 0 if len(state.shape) == 1 else 1
        state = np.concatenate(state, reward_weights, axis=axis)

    return state


def parse_replications(replications: Union[str, Tuple[int, ...]]) -> List[int]:
    if isinstance(replications, tuple):
        return list(replications)
    elif isinstance(replications, str):
        if "-" in replications:
            assert "," not in replications, "Cannot mix ranges and commas"
            start_str, stop_str = replications.split("-")
            start = int(start_str)
            stop = int(stop_str)
            return list(range(start, stop + 1))
        elif "," in replications:
            assert "-" not in replications, "Cannot mix ranges and commas"
            return [int(token) for token in replications.split(",")]
        else:
            return list(range(1, int(replications) + 1))
    elif isinstance(replications, int):
        return list(range(1, int(replications) + 1))
    else:
        raise ValueError(f"Unsupported replications string {replications}")


def save_reward(
    query_type: str,
    w_sampler,
    n_reward_samples: int,
    outdir: Path,
    true_delta: Optional[float] = None,
):
    np.save(
        outdir / "mean_reward.npy",
        make_mode_reward(query_type, w_sampler, n_reward_samples, true_delta),
    )


def update_inputs(a_inputs, b_inputs, inputs: Optional[np.ndarray], outdir: Path) -> np.ndarray:
    """Adds a new pair of input trajectories (a_inputs, b_inputs) to the inputs list and saves it."""
    inputs = append(inputs, np.stack([a_inputs, b_inputs]))
    np.save(outdir / "inputs.npy", inputs)
    return inputs


def append(a: Optional[np.ndarray], b: Union[np.ndarray, int], flat=False) -> np.ndarray:
    if isinstance(b, np.ndarray) and not flat:
        b = b.reshape((1, *b.shape))

    if a is None:
        if isinstance(b, np.ndarray):
            return b
        elif isinstance(b, int):
            return np.array([b])
    else:
        if isinstance(b, np.ndarray):
            return np.append(a, b, axis=0)
        elif isinstance(b, int):
            return np.append(a, b)


def load(file: Path, overwrite: bool, default: Any = None) -> Any:
    if overwrite:
        return default

    if file.exists():
        if ".pkl" in file.name:
            return pkl.load(open(file, "rb"))
        elif ".npy" in file.name:
            return np.load(file)
        else:
            raise ValueError(f"Filetype of {file} unrecognized")

    return default


def make_mode_reward(
    query_type: str, w_sampler, n_reward_samples: int, true_delta: Optional[float] = None
) -> np.ndarray:
    w_samples, _ = w_sampler.sample_given_delta(n_reward_samples, query_type, true_delta)
    mean_weight = np.mean(w_samples, axis=0)
    normalized_mean_weight = mean_weight / np.linalg.norm(mean_weight)
    return normalized_mean_weight


def make_td3_paths(td3_path: Path, replication_indices: List[int]) -> List[Path]:
    """ Selects the most recent policy in each replication folder. """
    paths = list()
    for i in replication_indices:
        replication_dir = td3_path / str(i)
        # Each replication folder contains directories whose names are timestamps
        children = [child.name for child in replication_dir.iterdir() if child.is_dir()]

        timestamp_dirs = []
        timestamps = []
        for child in children:
            try:
                timestamps.append(arrow.get(child))
                timestamp_dirs.append(child)
            except arrow.ParserError:
                logging.warning(f"Ignoring non-timestamp directory {child}")

        most_recent_child = timestamp_dirs[np.argmax(timestamps)]  # type: ignore
        td3_dir = replication_dir / most_recent_child
        # Within each dir is a set of best_X_actor, best_X_critic, etc files
        # We need to provide best_X to the path
        replication_files = list(td3_dir.iterdir())
        assert len(replication_files) > 0
        best_files = [child.name for child in replication_files if "best_" in child.name]
        if len(best_files) == 0:
            # Legacy pathway for runs that don't have a best
            logging.warning("Best model not found, falling back to last model.")
            model_files = [child.name for child in replication_files if "_critic" in child.name]
            model_file = model_files[0]
            prefix = model_file.split("_")[0]
        else:
            best_file = best_files[0]
            prefix = "_".join(best_file.split("_")[0:2])
        path = td3_dir / prefix
        paths.append(path)
    return paths


def make_reward_path(reward_path: Union[str, Path]):
    reward_path = Path(reward_path)

    if ".npy" in reward_path.name:
        reward_dir = reward_path.parent
        reward_name = reward_path.name
    else:
        reward_dir = reward_path
        reward_name = "true_reward.npy"

    reward_dir.mkdir(parents=True, exist_ok=True)

    return reward_dir, reward_name


def assert_nonempty(*arrs) -> None:
    for arr in arrs:
        assert len(arr) > 0


def assert_reward(
    reward: np.ndarray, use_equiv: bool, n_reward_features: int = 4, eps: float = 0.000_001
) -> None:
    """ Asserts the given array is might be a reward feature vector. """
    assert np.all(np.isfinite(reward))
    assert reward.shape == (n_reward_features + int(use_equiv),)
    assert abs(norm(reward) - 1) < eps


def assert_rewards(
    rewards: np.ndarray, use_equiv: bool, n_reward_features: int = 4, eps: float = 0.000_001
) -> None:
    assert np.all(np.isfinite(rewards))
    assert len(rewards.shape) == 2
    assert rewards.shape[1] == n_reward_features + int(
        use_equiv
    ), f"rewards.shape={rewards.shape}, n_reward_features={n_reward_features}, use_equiv={use_equiv}"
    norm_dist = abs(norm(rewards, axis=1) - 1)
    norm_errors = norm_dist > eps
    if np.any(norm_errors):
        logging.error("Some rewards are not normalized")
        indices = np.where(norm_errors)
        logging.error(f"Bad distances:\n{norm_dist[indices]}")
        logging.error(f"Bad rewards:\n{rewards[indices]}")
        logging.error(f"Bad indices:\n{indices}")
        assert not np.any(norm_errors)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """ Takes in a 2d array of row vectors and ensures each row vector has an L_2 norm of 1."""
    return (vectors.T / norm(vectors, axis=1)).T


def get_mean_reward(
    elicited_input_features: np.ndarray,
    elicited_preferences: np.ndarray,
    M: int,
    query_type: str,
    delta: float,
):
    n_features = elicited_input_features.shape[2]
    w_sampler = Sampler(n_features)
    for (a_phi, b_phi), preference in zip(elicited_input_features, elicited_preferences):
        w_sampler.feed(a_phi, b_phi, [preference])
    reward_samples, _ = w_sampler.sample_given_delta(M, query_type, delta)
    mean_reward = np.mean(reward_samples, axis=0)
    assert len(mean_reward.shape) == 1 and mean_reward.shape[0] == n_features
    return mean_reward


# Jank functions for directly modifying file output because I wrote a bug and don't want to re-run
# everything.


def flip_prefs(preferences_path: Path) -> None:
    preferences = np.load(preferences_path)
    preferences *= -1
    np.save(preferences_path, preferences)


def trim(n_questions: int, datadir: Path) -> None:
    datadir = Path(datadir)
    normals = np.load(datadir / "normals.npy")
    input_features = np.load(datadir / "input_features.npy")
    preferences = np.load(datadir / "preferences.npy")
    inputs = np.load(datadir / "inputs.npy")

    assert normals.shape[0] == input_features.shape[0]
    assert normals.shape[0] == preferences.shape[0]
    assert normals.shape[0] == inputs.shape[0]

    normals = normals[n_questions:]
    input_features = input_features[n_questions:]
    preferences = preferences[n_questions:]
    inputs = inputs[n_questions:]

    np.save(datadir / "normals.npy", normals)
    np.save(datadir / "input_features.npy", input_features)
    np.save(datadir / "preferences.npy", preferences)
    np.save(datadir / "inputs.npy", inputs)


def fix_flags(flags_path: Path):
    flags = pkl.load(open(flags_path, "rb"))
    if "equiv_size" not in flags.keys():
        flags["equiv_size"] = flags["delta"]
    pkl.dump(flags, open(flags_path, "wb"))


if __name__ == "__main__":
    fire.Fire()
