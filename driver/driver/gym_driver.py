import logging
from typing import Any, Tuple

import gym  # type: ignore
import numpy as np
from gym.spaces import Box  # type: ignore

from driver import car, dynamics, lane
from driver.world import World


class GymDriver(gym.Env):
    def __init__(self, reward: np.ndarray, horizon: int) -> None:
        self.reward = reward

        self.world = World()
        center_lane = lane.StraightLane([0.0, -1.0], [0.0, 1.0], 0.17)
        self.world.lanes += [center_lane, center_lane.shifted(1), center_lane.shifted(-1)]
        self.world.roads += [center_lane]
        self.world.fences += [center_lane.shifted(2), center_lane.shifted(-2)]
        self.dyn = dynamics.CarDynamics(0.1)
        self.robot = car.Car(self.dyn, [0.0, -0.3, np.pi / 2.0, 0.4], color="orange")
        self.human = car.Car(self.dyn, [0.17, 0.0, np.pi / 2.0, 0.41], color="white")
        self.world.cars.append(self.robot)
        self.world.cars.append(self.human)
        self.initial_state = [self.robot.x, self.human.x]

        self.observation_space = Box(low=-1 * np.ones(shape=(2, 4)), high=np.ones(shape=(2, 4)))
        self.action_space = Box(low=-1 * np.ones(shape=(2,)), high=np.ones((2,)))

        self.horizon = horizon
        self.timestep = 0

    def state(self) -> np.ndarray:
        return np.stack([self.robot.x, self.human.x])

    def get_human_action(self, t: int, max_t: int):
        if t < max_t // 5:
            return [0, self.initial_state[1][3]]
        elif t < 2 * max_t // 5:
            return [1.0, self.initial_state[1][3]]
        elif t < 3 * max_t // 5:
            return [-1.0, self.initial_state[1][3]]
        elif t < 4 * max_t // 5:
            return [0, self.initial_state[1][3] * 1.3]
        else:
            return [0, self.initial_state[1][3] * 1.3]

    @staticmethod
    def get_features(state: np.ndarray) -> np.ndarray:
        # staying in lane (higher is better)
        min_dist_to_lane = min(
            (state[0, 0] - 0.17) ** 2, (state[0, 0]) ** 2, (state[0, 0] + 0.17) ** 2
        )
        staying_in_lane: float = np.exp(-30 * min_dist_to_lane) / 0.15343634

        # keeping speed (lower is better)
        keeping_speed: float = (state[0, 3] - 1) ** 2 / 0.42202643

        # heading (higher is better)
        heading: float = np.sin(state[0, 2]) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance: float = (
            np.exp(
                -(7.0 * (state[0, 0] - state[1, 0]) ** 2 + 3.0 * (state[0, 1] - state[1, 1]) ** 2)
            )
            / 0.15258019
        )

        return np.array([staying_in_lane, keeping_speed, heading, collision_avoidance], dtype=float)

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        self.robot.u = action
        self.human.u = self.get_human_action(self.timestep, self.horizon)

        self.robot.move()
        self.human.move()

        state = self.state()
        reward = self.reward @ self.get_features(state)
        done = self.timestep >= self.horizon

        self.timestep += 1

        return state, reward, done, {}

    def reset(self) -> Any:
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]
        return self.state()

    def close(self):
        pass

    def seed(self, seed=None):
        logging.warning("Environment is deterministic")
        return

    def render(self, mode="human"):
        raise NotImplementedError
