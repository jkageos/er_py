"""Population evaluation for novelty search."""

import os
from typing import List, Tuple, Union

import numpy as np

# Suppress pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from task11.config import EnvironmentConfig, Task11Config, get_config
from task11.controller import MazeController

# Type alias for behavior descriptors (must match novelty_archive.py)
BehaviorType = Union[Tuple[float, float], Tuple[float, float, float, float]]


def compute_behavior_descriptor(
    trajectory: List[Tuple[float, float]],
    env_width: int,
    env_height: int,
) -> Tuple[float, float, float, float]:
    """Compute extended behavior descriptor.

    Returns:
        (final_x, final_y, coverage_x, coverage_y)
        - final_x, final_y are in PIXEL coordinates (not normalized)
        - coverage_x, coverage_y are normalized [0, 1]
    """
    if not trajectory:
        return (0.0, 0.0, 0.0, 0.0)

    final_x, final_y = trajectory[-1]

    # Compute trajectory spread (how much area was explored)
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]

    coverage_x = (max(xs) - min(xs)) / env_width
    coverage_y = (max(ys) - min(ys)) / env_height

    # Return final position in PIXELS (not normalized!)
    return (
        final_x,
        final_y,
        coverage_x,
        coverage_y,
    )


def evaluate_genome(
    genome: np.ndarray,
    env_config: EnvironmentConfig,
    n_hidden: int,
    n_hidden_layers: int,
    eval_steps: int,
    maze_file: str,
    extended_behavior: bool = False,
) -> Tuple[BehaviorType, List[Tuple[float, float]], bool]:
    """Evaluate a single genome.

    Args:
        genome: Neural network weights
        env_config: Environment configuration
        n_hidden: Hidden layer size
        n_hidden_layers: Number of hidden layers
        eval_steps: Number of evaluation steps
        maze_file: Path to maze file
        extended_behavior: Whether to compute extended behavior descriptor

    Returns:
        Tuple of (behavior_descriptor, trajectory, goal_reached)
    """
    # Import here to avoid circular imports and pygame issues in workers
    from task11.environment import MazeEnvironment

    env = MazeEnvironment(
        config=env_config,
        maze_file=maze_file,
        render=False,
        spawn_robot=True,
    )

    controller = MazeController(
        n_hidden=n_hidden,
        n_hidden_layers=n_hidden_layers,
        genome=genome,
        n_lidar_rays=env_config.n_lidar_rays,
    )

    env.reset(controller=controller)

    for _ in range(eval_steps):
        # step() returns False when goal is reached
        if not env.step():
            break

    final_pos = env.get_position()
    trajectory = env.get_trajectory()
    goal_reached = env.is_goal_reached()

    env.close()

    behavior: BehaviorType
    if extended_behavior:
        behavior = compute_behavior_descriptor(
            trajectory, env_config.width, env_config.height
        )
    else:
        behavior = final_pos

    return behavior, trajectory, goal_reached


class PopulationEvaluator:
    """Evaluate a population of genomes."""

    def __init__(self, config: Task11Config | None = None):
        """Initialize evaluator.

        Args:
            config: Task configuration (uses default if None)
        """
        self.config = config or get_config()
        self._env_config = self.config.environment
        self._evo_config = self.config.evolution
        self._parallel_config = self.config.parallel
        self._novelty_config = self.config.novelty
        self._maze_file = str(self.config.maze_file)

    def _evaluate_single(
        self, genome: np.ndarray
    ) -> Tuple[BehaviorType, List[Tuple[float, float]], bool]:
        """Evaluate a single genome."""
        return evaluate_genome(
            genome,
            self._env_config,
            self._evo_config.n_hidden,
            self._evo_config.n_hidden_layers,
            self._env_config.eval_steps,
            self._maze_file,
            extended_behavior=self._novelty_config.use_extended_behavior,
        )

    def evaluate(
        self, population: List[np.ndarray]
    ) -> Tuple[List[BehaviorType], List[List[Tuple[float, float]]], List[bool]]:
        """Evaluate entire population.

        Args:
            population: List of genomes

        Returns:
            Tuple of (behaviors, trajectories, goal_reached_flags)
        """
        if self._parallel_config.enabled and len(population) > 10:
            from utils.parallel import parallel_map

            results = parallel_map(
                self._evaluate_single,
                population,
                n_workers=self._parallel_config.n_workers,
            )
        else:
            results = [self._evaluate_single(g) for g in population]

        behaviors: List[BehaviorType] = [r[0] for r in results]
        trajectories: List[List[Tuple[float, float]]] = [r[1] for r in results]
        goal_reached: List[bool] = [r[2] for r in results]

        return behaviors, trajectories, goal_reached
