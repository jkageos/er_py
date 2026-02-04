"""Metrics and fitness computation for novelty search evaluation."""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode/evaluation."""

    final_position: Tuple[float, float]
    trajectory: List[Tuple[float, float]]
    goal_reached: bool
    steps_taken: int

    # Computed metrics
    distance_to_goal: float = 0.0
    path_length: float = 0.0
    exploration_area: float = 0.0

    def __post_init__(self):
        """Compute derived metrics."""
        if self.trajectory:
            self.path_length = self._compute_path_length()
            self.exploration_area = self._compute_exploration_area()

    def _compute_path_length(self) -> float:
        """Compute total path length traveled."""
        if len(self.trajectory) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i][0] - self.trajectory[i - 1][0]
            dy = self.trajectory[i][1] - self.trajectory[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def _compute_exploration_area(self) -> float:
        """Compute bounding box area of trajectory."""
        if not self.trajectory:
            return 0.0

        xs = [p[0] for p in self.trajectory]
        ys = [p[1] for p in self.trajectory]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return width * height


@dataclass
class GenerationMetrics:
    """Aggregated metrics for one generation."""

    generation: int

    # Novelty metrics
    avg_novelty: float = 0.0
    max_novelty: float = 0.0
    archive_size: int = 0

    # Coverage metrics
    coverage: float = 0.0
    cells_visited: int = 0

    # Goal metrics
    goals_reached: int = 0
    best_distance_to_goal: float = float("inf")
    avg_distance_to_goal: float = float("inf")

    # Exploration metrics
    avg_path_length: float = 0.0
    max_path_length: float = 0.0
    avg_exploration_area: float = 0.0

    # Diversity metrics
    position_spread: float = 0.0  # Std dev of final positions


@dataclass
class EvolutionHistory:
    """Complete history of evolution run."""

    generations: List[GenerationMetrics] = field(default_factory=list)

    # Best solutions found
    best_goal_genome: np.ndarray | None = None
    best_goal_generation: int | None = None
    best_novelty_genome: np.ndarray | None = None
    best_novelty_score: float = 0.0

    def add_generation(self, metrics: GenerationMetrics) -> None:
        """Add generation metrics."""
        self.generations.append(metrics)

    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        return [getattr(g, metric_name) for g in self.generations]

    @property
    def avg_novelty_history(self) -> List[float]:
        return self.get_metric_history("avg_novelty")

    @property
    def max_novelty_history(self) -> List[float]:
        return self.get_metric_history("max_novelty")

    @property
    def coverage_history(self) -> List[float]:
        return self.get_metric_history("coverage")

    @property
    def archive_size_history(self) -> List[int]:
        return [g.archive_size for g in self.generations]

    @property
    def goals_reached_history(self) -> List[int]:
        return [g.goals_reached for g in self.generations]

    @property
    def best_distance_history(self) -> List[float]:
        return self.get_metric_history("best_distance_to_goal")


def compute_generation_metrics(
    generation: int,
    behaviors: List[Tuple[float, ...]],
    trajectories: List[List[Tuple[float, float]]],
    goal_reached: List[bool],
    novelty_scores: List[float],
    archive_size: int,
    coverage: float,
    cells_visited: int,
    goal_pos: Tuple[float, float],
) -> GenerationMetrics:
    """Compute all metrics for a generation."""

    # Compute distances to goal
    distances = []
    for behavior in behaviors:
        dx = behavior[0] - goal_pos[0]
        dy = behavior[1] - goal_pos[1]
        distances.append(math.sqrt(dx * dx + dy * dy))

    # Compute path lengths
    path_lengths = []
    exploration_areas = []
    for traj in trajectories:
        if len(traj) >= 2:
            length = sum(
                math.sqrt(
                    (traj[i][0] - traj[i - 1][0]) ** 2
                    + (traj[i][1] - traj[i - 1][1]) ** 2
                )
                for i in range(1, len(traj))
            )
            path_lengths.append(length)

            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            exploration_areas.append(area)

    # Compute position spread (diversity)
    final_xs = [b[0] for b in behaviors]
    final_ys = [b[1] for b in behaviors]
    position_spread = math.sqrt(np.std(final_xs) ** 2 + np.std(final_ys) ** 2)

    return GenerationMetrics(
        generation=generation,
        avg_novelty=float(np.mean(novelty_scores)) if novelty_scores else 0.0,
        max_novelty=float(np.max(novelty_scores)) if novelty_scores else 0.0,
        archive_size=archive_size,
        coverage=coverage,
        cells_visited=cells_visited,
        goals_reached=sum(goal_reached),
        best_distance_to_goal=min(distances) if distances else float("inf"),
        avg_distance_to_goal=float(np.mean(distances)) if distances else float("inf"),
        avg_path_length=float(np.mean(path_lengths)) if path_lengths else 0.0,
        max_path_length=max(path_lengths) if path_lengths else 0.0,
        avg_exploration_area=float(np.mean(exploration_areas))
        if exploration_areas
        else 0.0,
        position_spread=position_spread,
    )


def compute_fitness_proxy(metrics: GenerationMetrics) -> Tuple[float, float]:
    """Compute fitness-like metrics for plotting.

    For novelty search, we use:
    - "Best fitness": combination of coverage and goal proximity
    - "Average fitness": average normalized distance improvement

    Returns:
        (best_fitness, avg_fitness) in [0, 1]
    """
    # Normalize distance (assuming max distance ~ 1500 for 1200x1000 env)
    max_distance = 1500.0

    # Best fitness: weighted combination
    coverage_weight = 0.4
    distance_weight = 0.4
    goal_weight = 0.2

    normalized_distance = 1.0 - min(1.0, metrics.best_distance_to_goal / max_distance)
    goal_bonus = 1.0 if metrics.goals_reached > 0 else 0.0

    best_fitness = (
        coverage_weight * metrics.coverage
        + distance_weight * normalized_distance
        + goal_weight * goal_bonus
    )

    # Average fitness
    avg_normalized_distance = 1.0 - min(
        1.0, metrics.avg_distance_to_goal / max_distance
    )
    avg_fitness = (
        coverage_weight * metrics.coverage + distance_weight * avg_normalized_distance
    )

    return best_fitness, avg_fitness
