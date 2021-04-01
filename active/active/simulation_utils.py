import logging
from typing import Optional, Tuple

import numpy as np
import scipy.optimize as opt  # type: ignore
from driver.car import LegacyPlanCar, LegacyPlannerCar
from driver.legacy.models import Driver  # type: ignore
from driver.legacy.simulator import Simulation  # type: ignore
from driver.world import ThreeLaneCarWorld

from active import algos


def make_opt_traj(
    reward: np.ndarray, start_state: Optional[np.ndarray] = None, n_planner_iter: int = 10
) -> np.ndarray:
    world = ThreeLaneCarWorld()
    if start_state is None:
        main_init = np.array([0.0, -0.3, np.pi / 2.0, 0.4])
        other_init = None
    else:
        assert start_state.shape == (2, 4)
        main_init, other_init = start_state[0], start_state[1]
    planner_car = LegacyPlannerCar(
        env=world,
        init_state=main_init,
        horizon=50,
        weights=reward,
        planner_args={"n_iter": n_planner_iter},
    )
    other_car = LegacyPlanCar(env=world, init_state=other_init)
    world.add_cars([planner_car, other_car])
    plan = planner_car._get_next_control().numpy()
    return plan


def get_simulated_feedback(
    simulation: Simulation,
    input_A: np.ndarray,
    input_B: np.ndarray,
    query_type: str,
    true_reward: np.ndarray,
    delta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    simulation.feed(input_A)
    phi_A = np.array(simulation.get_features())
    simulation.feed(input_B)
    phi_B = np.array(simulation.get_features())
    if query_type == "weak":
        # TODO(joschnei): Implement weak errors using delta. I think there's a model for this but I can't remember off hand.
        raise NotImplementedError("Simulated weak preferences not implemented.")
        if delta is None:
            raise ValueError("Must provide delta when using weak queries.")
    elif query_type == "strict":
        s = 1 if true_reward @ (phi_A - phi_B) > 0 else -1
    else:
        raise ValueError(f'query type {query_type} must be either "strict" or "weak"')
    return phi_A, phi_B, s


def get_feedback(simulation_object, input_A, input_B, query_type):
    simulation_object.feed(input_A)
    phi_A = np.array(simulation_object.get_features())
    simulation_object.feed(input_B)
    phi_B = np.array(simulation_object.get_features())
    s = -2
    while s == -2:
        if query_type == "weak":
            selection = input('A/B to watch, 1/2 to vote, 0 for "About Equal": ').lower()
        elif query_type == "strict":
            selection = input("A/B to watch, 1/2 to vote: ").lower()
        else:
            raise ValueError("There is no query type called " + query_type)
        if selection == "a":
            simulation_object.feed(input_A)
            simulation_object.watch(1)
        elif selection == "b":
            simulation_object.feed(input_B)
            simulation_object.watch(1)
        elif selection == "0" and query_type == "weak":
            s = 0
        elif selection == "1":
            s = 1
        elif selection == "2":
            s = -1
    return phi_A, phi_B, s


def create_env(task: str) -> Simulation:
    if task == "driver":
        return Driver()
    else:
        raise ValueError("There is no task called " + task)


def run_algo(criterion, simulation_object, w_samples, delta_samples, continuous: bool = False):
    if criterion == "information":
        return algos.information(simulation_object, w_samples, delta_samples, continuous)
    if criterion == "volume":
        return algos.volume(simulation_object, w_samples, delta_samples, continuous)
    elif criterion == "random":
        return algos.random(simulation_object)
    else:
        raise ValueError("There is no criterion called " + criterion)


def func(ctrl_array, *args):
    simulation_object = args[0]
    w = np.array(args[1])
    simulation_object.set_ctrl(ctrl_array)
    features = np.array(simulation_object.get_features())
    assert features.shape == (4,)
    assert w.shape == (4,)
    return -np.mean(features.dot(w))


def compute_best(simulation_object, w, iter_count=10) -> np.ndarray:
    u = simulation_object.ctrl_size
    lower_ctrl_bound = [x[0] for x in simulation_object.ctrl_bounds]
    upper_ctrl_bound = [x[1] for x in simulation_object.ctrl_bounds]
    opt_val = np.inf
    optimal_ctrl: Optional[np.ndarray] = None
    for _ in range(iter_count):
        temp_res = opt.fmin_l_bfgs_b(
            func,
            x0=np.random.uniform(low=lower_ctrl_bound, high=upper_ctrl_bound, size=(u)),
            args=(simulation_object, w),
            bounds=simulation_object.ctrl_bounds,
            approx_grad=True,
        )
        if temp_res[1] < opt_val:
            optimal_ctrl = temp_res[0]
            opt_val = temp_res[1]
    if optimal_ctrl is None:
        raise RuntimeError("No solution found.")
    logging.info(f"Optimal value={-opt_val}")
    return optimal_ctrl


def play(simulation_object, optimal_ctrl):
    simulation_object.set_ctrl(optimal_ctrl)
    keep_playing = "y"
    while keep_playing == "y":
        keep_playing = "u"
        simulation_object.watch(1)
        while keep_playing != "n" and keep_playing != "y":
            keep_playing = input("Again? [y/n]: ").lower()
    return optimal_ctrl
