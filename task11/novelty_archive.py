"""Novelty archive for behavior-based diversity search."""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Type alias for behavior descriptors
BehaviorType = Union[Tuple[float, float], Tuple[float, float, float, float]]


class NoveltyArchive:
    """Archive for storing novel behaviors with k-nearest neighbor novelty computation."""

    def __init__(
        self,
        env_width: int = 800,
        env_height: int = 600,
        k_nearest: int = 15,
        archive_threshold: float = 0.01,
        target_add_rate: float = 0.1,
        adapt_rate: float = 0.05,
        min_threshold: float = 0.005,
        max_threshold: float = 0.1,
        max_size: int = 10000,
    ):
        """Initialize novelty archive.

        Args:
            env_width: Environment width in pixels
            env_height: Environment height in pixels
            k_nearest: Number of nearest neighbors for novelty computation
            archive_threshold: Normalized threshold (fraction of diagonal) for adding to archive
            target_add_rate: Target fraction of population to add per generation
            adapt_rate: Rate of threshold adaptation
            min_threshold: Minimum normalized threshold
            max_threshold: Maximum normalized threshold
            max_size: Maximum archive size
        """
        self.env_width = env_width
        self.env_height = env_height
        self.k_nearest = k_nearest
        self.target_add_rate = target_add_rate
        self.adapt_rate = adapt_rate
        self.max_size = max_size

        # Calculate environment diagonal for scaling
        self.env_diagonal = math.sqrt(env_width**2 + env_height**2)

        # Store normalized thresholds (as fraction of diagonal)
        self._normalized_threshold = archive_threshold
        self._normalized_min = min_threshold
        self._normalized_max = max_threshold

        # Archive threshold in pixel space
        self.archive_threshold = self._normalized_threshold * self.env_diagonal
        self.min_threshold = self._normalized_min * self.env_diagonal
        self.max_threshold = self._normalized_max * self.env_diagonal

        # Main archive for novelty computation (stores pixel coordinates)
        self._archive_array: NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)

        # Result archive: tracks ALL unique cells visited
        self._grid_size = 50
        self._result_grid: NDArray[np.bool_] = np.zeros(
            (self._grid_size, self._grid_size), dtype=bool
        )

        # Store genomes along with behaviors for result archive
        self._genome_archive: List[Tuple[np.ndarray, Tuple[float, float]]] = []

        # Track best genome per grid cell (like GridArchive)
        self._cell_to_genome: dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}

        # Reachable cells mask
        self._reachable_mask: Optional[NDArray[np.bool_]] = None
        self._num_reachable_cells: int = self._grid_size * self._grid_size

        # Track recent add rates
        self._recent_add_rates: List[float] = []
        self._rate_window = 10

    def set_reachable_mask(
        self,
        maze_grid: List[List[dict]],
        wall_thickness: float,
        robot_radius: float = 10.0,
    ) -> None:
        """Set which grid cells are actually reachable (not walls)."""
        if not maze_grid or not maze_grid[0]:
            self._reachable_mask = np.ones(
                (self._grid_size, self._grid_size), dtype=bool
            )
            self._num_reachable_cells = self._grid_size * self._grid_size
            return

        maze_rows = len(maze_grid)
        maze_cols = len(maze_grid[0])
        margin = wall_thickness
        cell_width = (self.env_width - 2 * margin) / maze_cols
        cell_height = (self.env_height - 2 * margin) / maze_rows

        self._reachable_mask = np.ones((self._grid_size, self._grid_size), dtype=bool)
        grid_cell_w = self.env_width / self._grid_size
        grid_cell_h = self.env_height / self._grid_size
        effective_wall = wall_thickness / 2 + robot_radius

        for gy in range(self._grid_size):
            for gx in range(self._grid_size):
                px = (gx + 0.5) * grid_cell_w
                py = (gy + 0.5) * grid_cell_h

                # Check outer boundary
                if (
                    px < margin + effective_wall
                    or px > self.env_width - margin - effective_wall
                    or py < margin + effective_wall
                    or py > self.env_height - margin - effective_wall
                ):
                    self._reachable_mask[gy, gx] = False
                    continue

                # Determine maze cell
                maze_col = max(0, min(maze_cols - 1, int((px - margin) / cell_width)))
                maze_row = max(0, min(maze_rows - 1, int((py - margin) / cell_height)))

                cell_x = margin + maze_col * cell_width
                cell_y = margin + maze_row * cell_height
                local_x = px - cell_x
                local_y = py - cell_y

                walls = maze_grid[maze_row][maze_col]

                if walls.get("N", False) and local_y < effective_wall:
                    self._reachable_mask[gy, gx] = False
                elif walls.get("S", False) and local_y > cell_height - effective_wall:
                    self._reachable_mask[gy, gx] = False
                elif walls.get("W", False) and local_x < effective_wall:
                    self._reachable_mask[gy, gx] = False
                elif walls.get("E", False) and local_x > cell_width - effective_wall:
                    self._reachable_mask[gy, gx] = False

        self._num_reachable_cells = int(np.sum(self._reachable_mask))

    def __len__(self) -> int:
        return len(self._archive_array)

    def get_behaviors(self) -> List[Tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self._archive_array]

    def get_normalized_threshold(self) -> float:
        return self.archive_threshold / self.env_diagonal

    def compute_novelty_batch(self, behaviors: List[BehaviorType]) -> List[float]:
        """Compute novelty for all behaviors using k-nearest neighbors.

        Novelty is the average distance to k-nearest neighbors.
        Only uses the first two dimensions (x, y) for novelty computation.
        Behaviors should be in PIXEL coordinates.

        Returns:
            List of novelty scores (distances in pixels)
        """
        if not behaviors:
            return []

        # Extract only x, y coordinates (first 2 dimensions) - these should be in pixels
        pop_arr = np.array([(b[0], b[1]) for b in behaviors], dtype=np.float64)
        n_pop = len(pop_arr)

        # Combine current population with archive for comparison
        all_arr = (
            np.vstack([pop_arr, self._archive_array])
            if len(self._archive_array) > 0
            else pop_arr
        )

        # Compute pairwise distances (in pixels)
        diff = pop_arr[:, np.newaxis, :] - all_arr[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        novelty_scores = np.zeros(n_pop, dtype=np.float64)

        for i in range(n_pop):
            sorted_dists = np.sort(distances[i])
            # Skip self (distance 0)
            start_idx = 1 if sorted_dists[0] < 1e-10 else 0
            k = min(self.k_nearest, len(sorted_dists) - start_idx)
            if k > 0:
                novelty_scores[i] = np.mean(sorted_dists[start_idx : start_idx + k])

        return novelty_scores.tolist()

    def update(self, behaviors: List[BehaviorType], novelty_scores: List[float]) -> int:
        """Update archive with novel behaviors.

        Args:
            behaviors: List of behavior descriptors (x, y in pixels)
            novelty_scores: Corresponding novelty scores (distances in pixels)

        Returns:
            Number of behaviors added to archive
        """
        # Extract only x, y coordinates
        behaviors_xy = [(b[0], b[1]) for b in behaviors]
        behaviors_arr = np.array(behaviors_xy, dtype=np.float64)
        novelty_arr = np.array(novelty_scores, dtype=np.float64)

        # ALWAYS update result grid (for coverage tracking)
        self._update_result_grid(behaviors_arr)

        # Add to novelty archive if above threshold
        mask = novelty_arr >= self.archive_threshold
        num_added = int(np.sum(mask))

        if num_added > 0:
            new_behaviors = behaviors_arr[mask]
            self._archive_array = (
                new_behaviors
                if len(self._archive_array) == 0
                else np.vstack([self._archive_array, new_behaviors])
            )

        # Enforce max size
        if len(self._archive_array) > self.max_size:
            self._archive_array = self._archive_array[-self.max_size :]

        return num_added

    def _update_result_grid(self, behaviors: NDArray[np.float64]) -> None:
        """Update the coverage grid with new behaviors.

        Args:
            behaviors: Array of (x, y) positions in pixels
        """
        grid_x = np.clip(
            (behaviors[:, 0] / self.env_width * self._grid_size).astype(np.int32),
            0,
            self._grid_size - 1,
        )
        grid_y = np.clip(
            (behaviors[:, 1] / self.env_height * self._grid_size).astype(np.int32),
            0,
            self._grid_size - 1,
        )
        self._result_grid[grid_y, grid_x] = True

    def adapt_threshold(self, num_added: int, population_size: int) -> None:
        """Adapt archive threshold based on add rate."""
        actual_rate = num_added / population_size if population_size > 0 else 0
        self._recent_add_rates.append(actual_rate)
        if len(self._recent_add_rates) > self._rate_window:
            self._recent_add_rates.pop(0)

        smoothed_rate = np.mean(self._recent_add_rates)

        if smoothed_rate > self.target_add_rate * 1.5:
            # Adding too many - increase threshold
            self.archive_threshold *= 1 + self.adapt_rate
        elif smoothed_rate < self.target_add_rate * 0.5:
            # Adding too few - decrease threshold
            self.archive_threshold *= 1 - self.adapt_rate

        self.archive_threshold = float(
            np.clip(self.archive_threshold, self.min_threshold, self.max_threshold)
        )

    def compute_coverage(self) -> float:
        """Compute coverage as fraction of reachable cells visited."""
        if self._reachable_mask is not None:
            visited_reachable = np.sum(self._result_grid & self._reachable_mask)
            return (
                float(visited_reachable) / self._num_reachable_cells
                if self._num_reachable_cells > 0
                else 0.0
            )
        return float(np.sum(self._result_grid)) / (self._grid_size * self._grid_size)

    def get_num_cells_visited(self) -> int:
        return int(np.sum(self._result_grid))

    def get_num_reachable_cells(self) -> int:
        return self._num_reachable_cells

    def get_reachable_mask(self) -> Optional[NDArray[np.bool_]]:
        return self._reachable_mask

    def get_result_grid(self) -> NDArray[np.bool_]:
        """Get the result grid showing visited cells."""
        return self._result_grid.copy()

    def get_result_grid_positions(self) -> List[Tuple[float, float]]:
        """Get pixel positions of visited grid cells."""
        positions = []
        grid_cell_w = self.env_width / self._grid_size
        grid_cell_h = self.env_height / self._grid_size

        for gy in range(self._grid_size):
            for gx in range(self._grid_size):
                if self._result_grid[gy, gx]:
                    px = (gx + 0.5) * grid_cell_w
                    py = (gy + 0.5) * grid_cell_h
                    positions.append((px, py))

        return positions

    def update_with_genomes(
        self,
        behaviors: List[BehaviorType],
        novelty_scores: List[float],
        genomes: List[np.ndarray],
    ) -> int:
        """Update archive with novel behaviors and their genomes.

        Args:
            behaviors: List of behavior descriptors (x, y in pixels)
            novelty_scores: Corresponding novelty scores
            genomes: Corresponding genomes

        Returns:
            Number of behaviors added to archive
        """
        # Extract only x, y coordinates (in pixels)
        behaviors_xy = [(b[0], b[1]) for b in behaviors]
        behaviors_arr = np.array(behaviors_xy, dtype=np.float64)
        novelty_arr = np.array(novelty_scores, dtype=np.float64)

        # ALWAYS update result grid (for coverage tracking)
        self._update_result_grid(behaviors_arr)

        # Update genome archive (store best genome per cell)
        for behavior, novelty, genome in zip(behaviors, novelty_scores, genomes):
            gx = int(
                np.clip(
                    behavior[0] / self.env_width * self._grid_size,
                    0,
                    self._grid_size - 1,
                )
            )
            gy = int(
                np.clip(
                    behavior[1] / self.env_height * self._grid_size,
                    0,
                    self._grid_size - 1,
                )
            )
            cell = (gx, gy)

            # Keep genome with highest novelty per cell
            if (
                cell not in self._cell_to_genome
                or novelty > self._cell_to_genome[cell][1]
            ):
                self._cell_to_genome[cell] = (genome.copy(), novelty)

        # Add to novelty archive if above threshold
        mask = novelty_arr >= self.archive_threshold
        num_added = int(np.sum(mask))

        if num_added > 0:
            new_behaviors = behaviors_arr[mask]
            self._archive_array = (
                new_behaviors
                if len(self._archive_array) == 0
                else np.vstack([self._archive_array, new_behaviors])
            )

        # Enforce max size
        if len(self._archive_array) > self.max_size:
            self._archive_array = self._archive_array[-self.max_size :]

        return num_added

    def get_diverse_genomes(self, n: int = 10) -> List[np.ndarray]:
        """Get n diverse genomes from different cells."""
        if not self._cell_to_genome:
            return []

        # Get genomes spread across the space
        cells = list(self._cell_to_genome.keys())
        np.random.shuffle(cells)

        return [self._cell_to_genome[c][0] for c in cells[:n]]
