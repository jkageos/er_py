"""Fitness evaluation for the ring problem."""

from typing import List

from task09.ring_environment import RingAgent, RingEnvironment


def evaluate_fitness(
    weights: List[float],
    environment: RingEnvironment,
    num_trials: int = 10,
    steps_per_trial: int = 100,
    threshold: float = 0.0,
) -> float:
    """
    Evaluate fitness of a controller across multiple trials.

    Fitness is the average proportion of time spent in the left half.

    Args:
        weights: Neural network weights
        environment: Ring environment
        num_trials: Number of trials with different starting positions
        steps_per_trial: Number of steps per trial
        threshold: Action threshold for the agent

    Returns:
        Fitness value in [0, 1] representing proportion of time in left half
    """
    agent = RingAgent(weights, threshold)
    total_left_time = 0
    total_steps = 0

    for _ in range(num_trials):
        # Random starting position
        start_pos = environment.total_cells // 2 + (
            _ % (environment.total_cells // 2)
        )  # Start in right half for harder test
        time_in_left, _ = agent.simulate(environment, steps_per_trial, start_pos)

        total_left_time += time_in_left
        total_steps += steps_per_trial

    return total_left_time / total_steps
