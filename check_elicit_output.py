# A series of sanity tests for output from the elicitation process.
import logging
import pickle
from pathlib import Path

import fire  # type: ignore
import numpy as np

from simulation_utils import create_env
from utils import assert_normals, assert_reward, get_mean_reward, orient_normals


def make_normals(input_features: np.ndarray) -> np.ndarray:
    normals = input_features[:, 0] - input_features[:, 1]
    assert_normals(normals, False, input_features.shape[2])
    return normals


def make_input_features(inputs: np.ndarray, sim) -> np.ndarray:
    input_features = np.empty((inputs.shape[0], 2, sim.num_of_features))
    for i, (a, b) in enumerate(inputs):
        sim.feed(a)
        input_features[i, 0] = sim.get_features()

        sim.feed(b)
        input_features[i, 1] = sim.get_features()

    return input_features


def assert_input_feature_consistency(inputs: np.ndarray, input_features: np.ndarray, sim) -> None:
    assert np.all(make_input_features(inputs, sim) == input_features)


def assert_normal_consistency(input_features: np.ndarray, normals: np.ndarray) -> None:
    assert np.all(make_normals(input_features) == normals)


def assert_true_reward_consistency(
    normals: np.ndarray, preferences: np.ndarray, true_reward: np.ndarray
) -> None:
    oriented_normals = orient_normals(normals, preferences)
    assert np.all(oriented_normals @ true_reward > 0)


def main(datadir: Path) -> None:
    logging.basicConfig(level="INFO")

    datadir = Path(datadir)

    flags = pickle.load(open(datadir / "flags.pkl", "rb"))
    use_equiv = False
    sim = create_env(flags["task"])
    n_reward_features = sim.num_of_features

    # Raw trajectory inputs
    inputs = np.load(datadir / "inputs.npy")
    n_questions = inputs.shape[0]
    assert inputs.shape[1] == 2

    # Mean posterior reward
    mean_reward = np.load(datadir / "mean_reward.npy")
    logging.info(mean_reward)
    assert_reward(mean_reward, use_equiv, n_reward_features)

    # Reward featutures of the inptus
    input_features = np.load(datadir / "input_features.npy")
    n_questions = input_features.shape[0]
    assert input_features.shape == (n_questions, 2, n_reward_features), input_features.shape

    normals = np.load(datadir / "normals.npy")
    logging.info(f"There are {normals.shape[0]} questions")
    assert_normals(normals, use_equiv, n_reward_features)

    preferences = np.load(datadir / "preferences.npy")
    assert preferences.shape == (n_questions,)
    assert np.all((preferences == 1) | (preferences == -1))

    true_reward = np.load(datadir / "true_reward.npy")
    assert_reward(true_reward, use_equiv, n_reward_features)
    logging.info(true_reward)

    oriented_normals = orient_normals(normals, preferences)

    assert_input_feature_consistency(inputs, input_features, sim)
    assert_normal_consistency(input_features, normals)
    assert_true_reward_consistency(normals, preferences, true_reward)

    mean_accuracy = np.mean(oriented_normals @ mean_reward > 0)
    logging.info(f"Accuracy of mean reward function is {mean_accuracy}")


if __name__ == "__main__":
    fire.Fire(main)
