from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


class PlotManager:
    """Manage all plotting operations for evolutionary robotics tasks."""

    def __init__(self, plots_dir: str = "plots", dpi: int = 150):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        self.dpi = dpi

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_novelty_progress(
        self,
        avg_novelty: List[float],
        max_novelty: List[float],
        archive_size: List[int],
        save_name: str = "task3_novelty_progress.png",
    ):
        """Plot novelty search progress."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Novelty over generations
        axes[0].plot(avg_novelty, label="Average Novelty", linewidth=2, color="#2E86AB")
        axes[0].plot(max_novelty, label="Max Novelty", linewidth=2, color="#A23B72")
        axes[0].set_xlabel("Generation", fontsize=12)
        axes[0].set_ylabel("Novelty Score", fontsize=12)
        axes[0].set_title("Novelty Search Progress", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Archive size
        axes[1].plot(archive_size, color="#06A77D", linewidth=2)
        axes[1].set_xlabel("Generation", fontsize=12)
        axes[1].set_ylabel("Archive Size", fontsize=12)
        axes[1].set_title("Behavior Archive Growth", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved novelty progress plot to {save_path}")

    def plot_exploration_map(
        self,
        archive: List[Tuple[float, float]],
        width: int = 800,
        height: int = 600,
        save_name: str = "task3_exploration_map.png",
    ):
        """Plot distribution of explored end positions."""
        fig, ax = plt.subplots(figsize=(10, 7.5))

        archive_x = [pos[0] for pos in archive]
        archive_y = [pos[1] for pos in archive]

        scatter = ax.scatter(
            archive_x,
            archive_y,
            alpha=0.6,
            s=30,
            c=range(len(archive)),
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)
        ax.set_title(
            "Distribution of Explored End Positions", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Discovery Order", fontsize=10)

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved exploration map to {save_path}")

    def plot_fitness_progress(
        self,
        avg_fitness: List[float],
        best_fitness: List[float],
        save_name: str = "task4_fitness_progress.png",
    ):
        """Plot fitness evolution over generations."""
        fig, ax = plt.subplots(figsize=(10, 6))

        generations = range(1, len(avg_fitness) + 1)

        ax.plot(
            generations,
            avg_fitness,
            label="Average Fitness",
            linewidth=2,
            color="#2E86AB",
            alpha=0.8,
        )
        ax.plot(
            generations,
            best_fitness,
            label="Best Fitness",
            linewidth=2.5,
            color="#A23B72",
        )

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title(
            "Fitness Evolution - Autonomous Recharging Task",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved fitness progress to {save_path}")

    def plot_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        width: int = 800,
        height: int = 600,
        charge_area: Optional[Tuple[float, float, float, float]] = None,
        title: str = "Robot Trajectory",
        save_name: str = "trajectory.png",
    ):
        """
        Plot single robot trajectory.

        Args:
            trajectory: List of (x, y) positions
            width: Arena width
            height: Arena height
            charge_area: (x, y, w, h) for charging area rectangle
            title: Plot title
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 7.5))

        # Draw charging area if provided
        if charge_area is not None:
            x, y, w, h = charge_area
            rect = Rectangle(
                (x, y),
                w,
                h,
                facecolor="gray",
                alpha=0.3,
                edgecolor="black",
                linewidth=2,
                label="Charging Area",
            )
            ax.add_patch(rect)

        # Draw trajectory
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]

        # Color gradient along trajectory
        points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Convert to list of line segments for LineCollection
        segments_list = [segments[i] for i in range(len(segments))]

        lc = LineCollection(segments_list, cmap="plasma", linewidth=2)
        lc.set_array(np.linspace(0, 1, len(traj_x)))
        line = ax.add_collection(lc)

        # Start and end markers
        ax.scatter(
            traj_x[0],
            traj_y[0],
            c="green",
            s=150,
            marker="o",
            edgecolors="black",
            linewidth=2,
            label="Start",
            zorder=5,
        )
        ax.scatter(
            traj_x[-1],
            traj_y[-1],
            c="red",
            s=150,
            marker="X",
            edgecolors="black",
            linewidth=2,
            label="End",
            zorder=5,
        )

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add colorbar for time
        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label("Time Progress", fontsize=10)

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved trajectory to {save_path}")

    def plot_multiple_trajectories(
        self,
        trajectories: List[List[Tuple[float, float]]],
        start_positions: List[Tuple[float, float]],
        width: int = 800,
        height: int = 600,
        charge_area: Optional[Tuple[float, float, float, float]] = None,
        save_name: str = "task4_test_trajectories.png",
    ):
        """Plot multiple trajectories in subplots."""
        n_traj = len(trajectories)
        cols = 2
        rows = (n_traj + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        axes = axes.flatten() if n_traj > 1 else [axes]

        for idx, (traj, start_pos) in enumerate(zip(trajectories, start_positions)):
            ax = axes[idx]

            # Draw charging area
            if charge_area is not None:
                x, y, w, h = charge_area
                rect = Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor="gray",
                    alpha=0.3,
                    edgecolor="black",
                    linewidth=1.5,
                    label="Charging Area",
                )
                ax.add_patch(rect)

            # Draw trajectory
            traj_x = [p[0] for p in traj]
            traj_y = [p[1] for p in traj]

            ax.plot(traj_x, traj_y, "b-", linewidth=1.5, alpha=0.7)
            ax.scatter(
                traj_x[0],
                traj_y[0],
                c="green",
                s=100,
                label="Start",
                zorder=5,
                edgecolors="black",
                linewidth=1,
            )
            ax.scatter(
                traj_x[-1],
                traj_y[-1],
                c="red",
                s=100,
                label="End",
                zorder=5,
                edgecolors="black",
                linewidth=1,
            )

            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_xlabel("X Position (pixels)", fontsize=10)
            ax.set_ylabel("Y Position (pixels)", fontsize=10)
            ax.set_title(
                f"Start: ({int(start_pos[0])}, {int(start_pos[1])})",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_traj, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved multiple trajectories to {save_path}")

    def plot_maze_exploration(
        self,
        trajectories: List[List[Tuple[float, float]]],
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        width: int,
        height: int,
        maze_grid: List[List[dict]],
        wall_thickness: float,
        cell_width: float,
        cell_height: float,
        save_name: str = "task3_exploration_map.png",
    ):
        """Plot all exploration trajectories on the maze."""
        fig, ax = plt.subplots(figsize=(12, 9))

        # Draw maze walls
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        # Draw walls
        for row in range(len(maze_grid)):
            for col in range(len(maze_grid[0])):
                cell_x = wall_thickness + col * cell_width
                cell_y = wall_thickness + row * cell_height
                cell_walls = maze_grid[row][col]

                if cell_walls.get("N", False):
                    ax.plot(
                        [cell_x, cell_x + cell_width],
                        [cell_y, cell_y],
                        "k-",
                        linewidth=1,
                    )
                if cell_walls.get("S", False):
                    ax.plot(
                        [cell_x, cell_x + cell_width],
                        [cell_y + cell_height, cell_y + cell_height],
                        "k-",
                        linewidth=1,
                    )
                if cell_walls.get("E", False):
                    ax.plot(
                        [cell_x + cell_width, cell_x + cell_width],
                        [cell_y, cell_y + cell_height],
                        "k-",
                        linewidth=1,
                    )
                if cell_walls.get("W", False):
                    ax.plot(
                        [cell_x, cell_x],
                        [cell_y, cell_y + cell_height],
                        "k-",
                        linewidth=1,
                    )

        # Draw outer boundary
        ax.plot([0, width], [0, 0], "k-", linewidth=2)
        ax.plot([0, width], [height, height], "k-", linewidth=2)
        ax.plot([0, 0], [0, height], "k-", linewidth=2)
        ax.plot([width, width], [0, height], "k-", linewidth=2)

        # Plot all trajectories with alpha transparency
        for i, trajectory in enumerate(trajectories):
            if not trajectory:
                continue
            traj_x = [p[0] for p in trajectory]
            traj_y = [p[1] for p in trajectory]
            ax.plot(traj_x, traj_y, alpha=0.15, linewidth=0.5, color="blue")

        # Mark start position
        ax.scatter(
            *start_pos,
            c="green",
            s=200,
            marker="o",
            edgecolors="darkgreen",
            linewidth=2,
            label="Start",
            zorder=5,
        )

        # Mark goal position
        ax.scatter(
            *goal_pos,
            c="red",
            s=200,
            marker="*",
            edgecolors="darkred",
            linewidth=2,
            label="Goal",
            zorder=5,
        )

        # Mark final positions
        final_positions = [traj[-1] for traj in trajectories if traj]
        if final_positions:
            final_x = [p[0] for p in final_positions]
            final_y = [p[1] for p in final_positions]
            ax.scatter(
                final_x,
                final_y,
                c="orange",
                s=50,
                alpha=0.6,
                edgecolors="darkorange",
                linewidth=0.5,
                label="Final Positions",
                zorder=4,
            )

        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)
        ax.set_title(
            "All Exploration Trajectories Across Generations",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved maze exploration map to {save_path}")
