from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Optional, cast

import numpy as np

from utils import load


class GeometricSearch:
    """Searches for a parameter with unknown bounds in the following way:
    1. Search geometrically (*= base) in each direction until you've overshot the goal
    2. Once you have established bounds, take the next value to be the geometric mean of the bounds.
    """

    def __init__(self, start: float, base: float = 10.0) -> None:
        self.value = start
        self.base = base
        self.min = start
        self.max = start

    def __call__(self, low: bool) -> float:
        if not low:
            self.min = min(self.min, self.value)
            if self.value < self.max:
                # If we already found a max we don't want to go above, pick a value between
                # the current covariance and the max
                self.value = np.sqrt(self.value * self.max)
            else:
                # Otherwise, grow geometrically
                self.value *= self.base
        else:
            self.max = max(self.max, self.value)
            if self.value > self.min:
                self.value = np.sqrt(self.value * self.min)
            else:
                self.value /= self.base
        return self.value


@dataclass
class Test:
    rewards: np.ndarray
    alignment: np.ndarray
    mean_alignment: float


class TestRewardSearch:
    def __init__(
        self,
        epsilon: float,
        cov_search: GeometricSearch,
        max_attempts: int,
        outdir: Path,
        new_rewards: Callable[[float], np.ndarray],
        get_alignment: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.epsilon: Final[float] = epsilon
        self.best_test: Optional[Test] = None
        self.last_test: Optional[Test] = None
        self.cov_search: Final[GeometricSearch] = cov_search
        self.attempt = 0
        self.max_attempts: Final[int] = max_attempts
        self.outdir = outdir

        self.new_rewards = new_rewards
        self.get_alignment = get_alignment

    def run(self) -> Test:
        while not self.is_done():
            self.attempt += 1

            cov = (
                self.cov_search(low=self.last_test.mean_alignment < 0.45)
                if self.last_test is not None
                else self.cov_search.value
            )
            if not np.isfinite(cov) or cov <= 0.0 or cov >= 100.0:
                if self.best_test is not None:
                    logging.warning(
                        f"cov={cov}, using best try with mean_alignment={self.best_test.mean_alignment}."
                    )
                else:
                    logging.warning(f"First cov={cov}, use inital covariance between 0 and 100.")
                # TODO(joschnei): Break is a code smell
                break

            self.last_test = self.make_test(cov=cov)

            logging.info(
                f"attempt={self.attempt} of {self.max_attempts}, mean_alignment={self.last_test.mean_alignment}"
            )

            if self.best_test is None or np.abs(self.last_test.mean_alignment - 0.5) < np.abs(
                self.best_test.mean_alignment - 0.5
            ):
                self.best_test = self.last_test

            pickle.dump(self, (self.outdir / "search.pkl").open("wb"))
            logging.debug("Dumped search")

        assert self.best_test is not None

        if self.attempt == self.max_attempts:
            logging.warning(
                f"Ran out of attempts, using test with mean_alignment={self.best_test.mean_alignment}"
            )

        return self.best_test

    def make_test(self, cov: float) -> Test:
        test_rewards = self.new_rewards(cov=cov)
        alignment = self.get_alignment(test_rewards=test_rewards)
        mean_align = cast(float, np.mean(alignment))
        return Test(rewards=test_rewards, alignment=alignment, mean_alignment=mean_align)

    @staticmethod
    def load(epsilon: float, path: Path, overwrite: bool = False) -> Optional[TestRewardSearch]:
        search: TestRewardSearch = load(path, overwrite)
        if search is None or search.epsilon != epsilon:
            return None
        return search

    def is_done(self) -> bool:
        if self.attempt == self.max_attempts:
            return True
        if (
            self.best_test is not None
            and self.best_test.mean_alignment > 0.45
            and self.best_test.mean_alignment < 0.55
        ):
            return True
        return False

    def __getstate__(self):
        """ Remove function arguments from pickle. """
        state = self.__dict__.copy()

        del state["new_rewards"]
        del state["get_alignment"]
        return state
