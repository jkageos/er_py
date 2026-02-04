"""Visualization for Task 11: Maze Exploration."""

import argparse
import pickle
from pathlib import Path

from task11.controller import MazeController
from task11.environment import MazeEnvironment
from utils.config_loader import ConfigLoader
from utils.plotting import PlotManager


class Visualizer:
    """Visualize best evolved controllers for Task 11."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader(config_path)

        # Get configurations
        self.task_config = self.config.get_task11_config()
        self.global_config = self.config.get_global_config()

        # Setup plot manager
        plots_dir = self.task_config.get("output", {}).get("plots_dir", "plots")
        dpi = self.global_config.get("dpi", 150)
        self.plotter = PlotManager(plots_dir, dpi)

    def visualize_task11(
        self, genome_path: str = "results/task11_best_genome.pkl"
    ) -> None:
        """Visualize Task 11 best controller."""
        print("\n" + "=" * 80)
        print("VISUALIZING TASK 11: Maze Exploration with Novelty Search")
        print("=" * 80)

        # Load genome
        genome_file = Path(genome_path)
        if not genome_file.exists():
            print(f"Error: Genome file not found at {genome_path}")
            return

        with open(genome_file, "rb") as f:
            best_genome = pickle.load(f)

        # Get environment parameters from config
        from task11.config import load_config

        config = load_config()

        # Get results directory for maze file
        results_dir = self.task_config.get("output", {}).get("results_dir", "results")
        maze_file = Path(results_dir) / "maze.pkl"

        # Create environment with rendering
        env = MazeEnvironment(
            config=config.environment,
            maze_file=maze_file,
            render=True,
        )

        # Create controller
        controller = MazeController(
            n_hidden=config.evolution.n_hidden,
            n_hidden_layers=config.evolution.n_hidden_layers,
            genome=best_genome,
        )

        # Run simulation and collect trajectory
        env.reset(controller=controller)
        controller.reset()

        trajectory = []
        eval_steps = config.environment.eval_steps

        print(f"\nRunning best controller for {eval_steps} steps...")
        for _ in range(eval_steps):
            env.step()
            env.render()
            pos = env.get_position()
            trajectory.append(pos)
            if env.screen is None:
                break

        env.close()

        # Plot trajectory
        print("\nGenerating trajectory plot...")
        self.plotter.plot_trajectory(
            trajectory=trajectory,
            width=config.environment.width,
            height=config.environment.height,
            charge_area=None,
            title="Best Novelty Search Controller - Maze Exploration",
            save_name="task11_best_trajectory.png",
        )

        print("\nTask 11 visualization complete!")

    def visualize_all(self) -> None:
        """Visualize all tasks."""
        self.visualize_task11()
        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("=" * 80)


def main() -> None:
    """Main visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize evolved controllers for Task Sheet 11"
    )
    parser.add_argument(
        "--task",
        choices=["11", "all"],
        default="all",
        help="Which task to visualize (default: all)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--genome",
        default="results/task11_best_genome.pkl",
        help="Path to best genome",
    )

    args = parser.parse_args()

    # Create visualizer
    viz = Visualizer(config_path=args.config)

    # Run visualization
    if args.task == "11":
        viz.visualize_task11(args.genome)
    else:
        viz.visualize_all()


if __name__ == "__main__":
    main()
