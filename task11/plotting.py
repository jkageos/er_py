"""Specialized plotting for Task 11 novelty search."""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from task11.metrics import EvolutionHistory, compute_fitness_proxy


class Task11Plotter:
    """Plotting utilities for Task 11 evolution results."""

    def __init__(self, output_dir: str = "plots", dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_fitness_progress(
        self,
        history: EvolutionHistory,
        save_name: str = "task11_fitness_progress.png",
    ) -> Path:
        """Plot fitness-like metrics over generations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        generations = list(range(1, len(history.generations) + 1))

        # Compute fitness proxies
        best_fitness = []
        avg_fitness = []
        for gen_metrics in history.generations:
            best, avg = compute_fitness_proxy(gen_metrics)
            best_fitness.append(best)
            avg_fitness.append(avg)

        # Plot 1: Fitness proxy
        ax = axes[0, 0]
        ax.plot(
            generations,
            best_fitness,
            label="Best Fitness",
            linewidth=2,
            color="#A23B72",
        )
        ax.plot(
            generations,
            avg_fitness,
            label="Average Fitness",
            linewidth=2,
            color="#2E86AB",
            alpha=0.8,
        )
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Fitness Score", fontsize=11)
        ax.set_title(
            "Fitness Progress (Coverage + Goal Proximity)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Coverage over time
        ax = axes[0, 1]
        coverage = history.coverage_history
        ax.plot(generations, [c * 100 for c in coverage], linewidth=2, color="#06A77D")
        ax.fill_between(
            generations, 0, [c * 100 for c in coverage], alpha=0.2, color="#06A77D"
        )
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Coverage (%)", fontsize=11)
        ax.set_title("Maze Coverage Over Time", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Plot 3: Distance to goal
        ax = axes[1, 0]
        best_distances = history.best_distance_history
        ax.plot(generations, best_distances, linewidth=2, color="#E84855")
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Distance to Goal (pixels)", fontsize=11)
        ax.set_title("Best Distance to Goal", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Mark when goal was first reached
        if history.best_goal_generation is not None:
            ax.axvline(
                x=history.best_goal_generation,
                color="green",
                linestyle="--",
                linewidth=2,
            )
            ax.annotate(
                f"Goal reached!\nGen {history.best_goal_generation}",
                xy=(history.best_goal_generation, min(best_distances)),
                xytext=(10, 30),
                textcoords="offset points",
                fontsize=10,
                color="green",
                arrowprops=dict(arrowstyle="->", color="green"),
            )

        # Plot 4: Goals reached per generation
        ax = axes[1, 1]
        goals = history.goals_reached_history
        ax.bar(generations, goals, color="#7B2CBF", alpha=0.7)
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Goals Reached", fontsize=11)
        ax.set_title("Goal Reaches Per Generation", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved fitness progress plot to {save_path}")
        return save_path

    def plot_novelty_metrics(
        self,
        history: EvolutionHistory,
        save_name: str = "task11_novelty_metrics.png",
    ) -> Path:
        """Plot novelty search specific metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        generations = list(range(1, len(history.generations) + 1))

        # Plot 1: Novelty scores
        ax = axes[0, 0]
        ax.plot(
            generations,
            history.avg_novelty_history,
            label="Average Novelty",
            linewidth=2,
            color="#2E86AB",
        )
        ax.plot(
            generations,
            history.max_novelty_history,
            label="Max Novelty",
            linewidth=2,
            color="#A23B72",
        )
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Novelty Score (pixels)", fontsize=11)
        ax.set_title("Novelty Scores Over Time", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Archive growth
        ax = axes[0, 1]
        ax.plot(generations, history.archive_size_history, linewidth=2, color="#06A77D")
        ax.fill_between(
            generations, 0, history.archive_size_history, alpha=0.2, color="#06A77D"
        )
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Archive Size", fontsize=11)
        ax.set_title("Behavior Archive Growth", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Plot 3: Position spread (diversity)
        ax = axes[1, 0]
        spreads = [g.position_spread for g in history.generations]
        ax.plot(generations, spreads, linewidth=2, color="#E84855")
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Position Spread (pixels)", fontsize=11)
        ax.set_title(
            "Population Diversity (Position Spread)", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Plot 4: Exploration metrics
        ax = axes[1, 1]
        path_lengths = [g.avg_path_length for g in history.generations]
        max_paths = [g.max_path_length for g in history.generations]
        ax.plot(
            generations,
            path_lengths,
            label="Average Path Length",
            linewidth=2,
            color="#2E86AB",
        )
        ax.plot(
            generations,
            max_paths,
            label="Max Path Length",
            linewidth=2,
            color="#A23B72",
            alpha=0.7,
        )
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Path Length (pixels)", fontsize=11)
        ax.set_title("Exploration Path Lengths", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved novelty metrics plot to {save_path}")
        return save_path

    def plot_best_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        goal_reached: bool,
        width: int,
        height: int,
        maze_grid: Optional[List[List[dict]]] = None,
        wall_thickness: float = 8.0,
        save_name: str = "task11_best_trajectory.png",
    ) -> Path:
        """Plot the best controller's trajectory."""
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert Y axis
        ax.set_aspect("equal")

        # Draw maze walls if provided
        if maze_grid:
            self._draw_maze_walls(ax, maze_grid, width, height, wall_thickness)

        # Draw trajectory with color gradient
        if len(trajectory) > 1:
            traj_x = [p[0] for p in trajectory]
            traj_y = [p[1] for p in trajectory]

            # Create segments for coloring
            points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Convert to list of segments for LineCollection
            segments_list = [segments[i] for i in range(len(segments))]
            lc = LineCollection(segments_list, cmap="plasma", linewidth=2.5)
            lc.set_array(np.linspace(0, 1, len(traj_x)))
            line = ax.add_collection(lc)

            # Colorbar
            cbar = plt.colorbar(line, ax=ax, shrink=0.8)
            cbar.set_label("Time Progress", fontsize=11)

        # Draw start marker
        ax.scatter(
            start_pos[0],
            start_pos[1],
            c="green",
            s=200,
            marker="o",
            edgecolors="darkgreen",
            linewidth=2,
            label="Start",
            zorder=5,
        )

        # Draw goal zone
        goal_color = "lightgreen" if goal_reached else "lightcoral"
        goal_circle = Circle(goal_pos, 40, color=goal_color, alpha=0.5)
        ax.add_patch(goal_circle)
        ax.scatter(
            goal_pos[0],
            goal_pos[1],
            c="red" if not goal_reached else "green",
            s=200,
            marker="*",
            edgecolors="darkred" if not goal_reached else "darkgreen",
            linewidth=2,
            label="Goal",
            zorder=5,
        )

        # Draw final position
        if trajectory:
            final = trajectory[-1]
            ax.scatter(
                final[0],
                final[1],
                c="blue",
                s=150,
                marker="X",
                edgecolors="darkblue",
                linewidth=2,
                label="Final",
                zorder=5,
            )

        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)

        # Use text instead of emoji to avoid font issues
        title = "Best Controller Trajectory"
        if goal_reached:
            title += " - GOAL REACHED!"
        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
            color="green" if goal_reached else "black",
        )

        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.2)

        # Add stats box
        if trajectory:
            path_length = self._compute_path_length(trajectory)
            goal_status = "YES" if goal_reached else "No"
            stats_text = f"Path length: {path_length:.0f} px\nSteps: {len(trajectory)}\nGoal reached: {goal_status}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved best trajectory plot to {save_path}")
        return save_path

    def _draw_maze_walls(
        self,
        ax,
        maze_grid: List[List[dict]],
        width: int,
        height: int,
        wall_thickness: float,
    ) -> None:
        """Draw maze walls on the axes."""
        maze_rows = len(maze_grid)
        maze_cols = len(maze_grid[0])
        margin = wall_thickness
        cell_width = (width - 2 * margin) / maze_cols
        cell_height = (height - 2 * margin) / maze_rows

        wall_color = "#404040"

        # Outer boundary
        ax.plot([0, width], [0, 0], color=wall_color, linewidth=2)
        ax.plot([0, width], [height, height], color=wall_color, linewidth=2)
        ax.plot([0, 0], [0, height], color=wall_color, linewidth=2)
        ax.plot([width, width], [0, height], color=wall_color, linewidth=2)

        # Internal walls
        for row in range(maze_rows):
            for col in range(maze_cols):
                cell = maze_grid[row][col]
                cell_x = margin + col * cell_width
                cell_y = margin + row * cell_height

                if cell.get("S", False) and row < maze_rows - 1:
                    y = cell_y + cell_height
                    ax.plot(
                        [cell_x, cell_x + cell_width],
                        [y, y],
                        color=wall_color,
                        linewidth=1.5,
                    )

                if cell.get("E", False) and col < maze_cols - 1:
                    x = cell_x + cell_width
                    ax.plot(
                        [x, x],
                        [cell_y, cell_y + cell_height],
                        color=wall_color,
                        linewidth=1.5,
                    )

    def _compute_path_length(self, trajectory: List[Tuple[float, float]]) -> float:
        """Compute total path length."""
        if len(trajectory) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def plot_exploration_map(
        self,
        trajectories: List[List[Tuple[float, float]]],
        width: int,
        height: int,
        maze_grid: Optional[List[List[dict]]] = None,
        wall_thickness: float = 8.0,
        save_name: str = "task11_exploration_map.png",
    ) -> Path:
        """Plot distribution of explored end positions (scatter plot)."""
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert Y axis
        ax.set_aspect("equal")

        # Draw maze walls if provided
        if maze_grid:
            self._draw_maze_walls(ax, maze_grid, width, height, wall_thickness)

        # Collect final positions from all trajectories
        final_positions = []
        for traj in trajectories:
            if traj:
                final_positions.append(traj[-1])

        if final_positions:
            final_x = [p[0] for p in final_positions]
            final_y = [p[1] for p in final_positions]

            # Scatter plot with color by order (discovery time)
            scatter = ax.scatter(
                final_x,
                final_y,
                c=range(len(final_positions)),
                cmap="viridis",
                s=30,
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
            )

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label("Discovery Order", fontsize=11)

        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)
        ax.set_title(
            f"Explored End Positions ({len(final_positions)} total)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved exploration map to {save_path}")
        return save_path
