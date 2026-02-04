"""Main entry point for running novelty search evolution."""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np

from task11.config import get_config, load_config
from task11.controller import MazeController
from task11.environment import MazeEnvironment
from task11.evolution import NoveltySearchEvolution
from task11.plotting import Task11Plotter


def run_evolution(config_path: str | None = None) -> tuple[bool, Path | None]:
    """Run the novelty search evolution.

    Args:
        config_path: Optional path to config file

    Returns:
        Tuple of (was_interrupted, best_genome_path)
    """
    print("Task 11: Novelty Search - Maze Exploration")
    print("=" * 80)

    config = load_config(config_path) if config_path else get_config()
    evolution = NoveltySearchEvolution(config)
    best_genome, _ = evolution.run()

    genome_path = config.genome_file if best_genome is not None else None
    return evolution.interrupted, genome_path


def evaluate_genome(
    genome: np.ndarray,
    config,
    render: bool = False,
) -> Tuple[List[Tuple[float, float]], bool]:
    """Evaluate a genome and return its trajectory.

    Args:
        genome: Controller genome
        config: Task11Config
        render: Whether to render during evaluation

    Returns:
        (trajectory, goal_reached)
    """
    env = MazeEnvironment(
        config=config.environment,
        maze_file=config.maze_file,
        render=render,
        spawn_robot=True,
    )

    controller = MazeController(
        n_hidden=config.evolution.n_hidden,
        n_hidden_layers=config.evolution.n_hidden_layers,
        genome=genome,
        n_lidar_rays=config.environment.n_lidar_rays,
    )

    env.reset(controller=controller)

    trajectory: List[Tuple[float, float]] = []
    for _ in range(config.environment.eval_steps):
        if not env.step():
            break

        if render:
            env.render()

        trajectory.append(env.get_position())

        if render and env.screen is None:
            break

    goal_reached = env.is_goal_reached()
    env.close()

    return trajectory, goal_reached


def test_best_controller(
    genome_path: Path | None = None,
    config_path: str | None = None,
    render: bool = False,
) -> None:
    """Test and visualize the best evolved controller."""
    config = load_config(config_path) if config_path else get_config()

    path = genome_path or config.genome_file
    if not path.exists():
        print(f"No genome found at {path}")
        return

    with open(path, "rb") as f:
        best_genome = pickle.load(f)

    print("\nTesting best controller...")
    print(f"  Max speed: {config.environment.max_speed:.1f} pixels/second")
    print(f"  Eval steps: {config.environment.eval_steps} steps")

    # Evaluate and get trajectory
    trajectory, goal_reached = evaluate_genome(best_genome, config, render=render)

    print("\nResults:")
    print(f"  Final position: ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f})")
    print(f"  Steps taken: {len(trajectory)}")
    if goal_reached:
        print("  ** GOAL REACHED!")
    else:
        print("  Goal not reached")

    # Save trajectory data for future use
    trajectory_file = Path(config.output.results_dir) / "best_trajectory.pkl"
    with open(trajectory_file, "wb") as f:
        pickle.dump(
            {
                "trajectory": trajectory,
                "goal_reached": goal_reached,
                "genome": best_genome,
            },
            f,
        )
    print(f"\nSaved trajectory data to {trajectory_file}")

    # Load maze grid for plotting
    maze_file = config.maze_file
    if maze_file.exists():
        with open(maze_file, "rb") as f:
            maze_grid = pickle.load(f)
    else:
        maze_grid = None

    # Calculate positions
    margin = config.environment.wall_thickness
    cell_width = (config.environment.width - 2 * margin) / config.environment.maze_cols
    cell_height = (
        config.environment.height - 2 * margin
    ) / config.environment.maze_rows

    start_pos = (margin + cell_width / 2, margin + cell_height / 2)
    goal_pos = (
        config.environment.width - margin - cell_width / 2,
        config.environment.height - margin - cell_height / 2,
    )

    # Create trajectory plot
    plotter = Task11Plotter(config.output.plots_dir)
    plotter.plot_best_trajectory(
        trajectory=trajectory,
        start_pos=start_pos,
        goal_pos=goal_pos,
        goal_reached=goal_reached,
        width=config.environment.width,
        height=config.environment.height,
        maze_grid=maze_grid,
        wall_thickness=config.environment.wall_thickness,
        save_name="task11_best_trajectory.png",
    )


def plot_saved_trajectory(
    config_path: str | None = None,
) -> None:
    """Plot trajectory from saved data (no re-evaluation needed)."""
    config = load_config(config_path) if config_path else get_config()

    # Try to load saved trajectory
    trajectory_file = Path(config.output.results_dir) / "best_trajectory.pkl"

    if trajectory_file.exists():
        print(f"Loading saved trajectory from {trajectory_file}")
        with open(trajectory_file, "rb") as f:
            data = pickle.load(f)
        trajectory = data["trajectory"]
        goal_reached = data["goal_reached"]
    else:
        # Fall back to re-evaluating the genome
        print("No saved trajectory found, re-evaluating best genome...")
        genome_file = config.genome_file
        if not genome_file.exists():
            print(f"No genome found at {genome_file}")
            print("Please run evolution first: python -m task11.run_evolution")
            return

        with open(genome_file, "rb") as f:
            best_genome = pickle.load(f)

        trajectory, goal_reached = evaluate_genome(best_genome, config, render=False)

        # Save for future use
        with open(trajectory_file, "wb") as f:
            pickle.dump(
                {
                    "trajectory": trajectory,
                    "goal_reached": goal_reached,
                    "genome": best_genome,
                },
                f,
            )
        print(f"Saved trajectory data to {trajectory_file}")

    # Load maze grid
    maze_file = config.maze_file
    if maze_file.exists():
        with open(maze_file, "rb") as f:
            maze_grid = pickle.load(f)
    else:
        maze_grid = None

    # Calculate positions
    margin = config.environment.wall_thickness
    cell_width = (config.environment.width - 2 * margin) / config.environment.maze_cols
    cell_height = (
        config.environment.height - 2 * margin
    ) / config.environment.maze_rows

    start_pos = (margin + cell_width / 2, margin + cell_height / 2)
    goal_pos = (
        config.environment.width - margin - cell_width / 2,
        config.environment.height - margin - cell_height / 2,
    )

    # Create trajectory plot
    plotter = Task11Plotter(config.output.plots_dir)
    plotter.plot_best_trajectory(
        trajectory=trajectory,
        start_pos=start_pos,
        goal_pos=goal_pos,
        goal_reached=goal_reached,
        width=config.environment.width,
        height=config.environment.height,
        maze_grid=maze_grid,
        wall_thickness=config.environment.wall_thickness,
        save_name="task11_best_trajectory.png",
    )

    # Also plot exploration map if trajectories are saved
    trajectories_file = Path(config.output.results_dir) / "all_trajectories.pkl"
    if trajectories_file.exists():
        print(f"Loading saved trajectories from {trajectories_file}")
        with open(trajectories_file, "rb") as f:
            all_trajectories = pickle.load(f)

        plotter.plot_exploration_map(
            all_trajectories,
            config.environment.width,
            config.environment.height,
            maze_grid=maze_grid,
            wall_thickness=config.environment.wall_thickness,
            save_name="task11_exploration_map.png",
        )

    print("\nTrajectory stats:")
    print(f"  Steps: {len(trajectory)}")
    print(f"  Final position: ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f})")
    print(f"  Goal reached: {'Yes' if goal_reached else 'No'}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 11 Novelty Search Evolution")
    parser.add_argument(
        "--test-only", action="store_true", help="Only test existing genome"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot saved trajectory (no evaluation)",
    )
    parser.add_argument("--render", action="store_true", help="Render during testing")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--genome", type=str, default=None, help="Path to genome file")

    args = parser.parse_args()

    if args.plot_only:
        plot_saved_trajectory(args.config)
    elif args.test_only:
        genome_path = Path(args.genome) if args.genome else None
        test_best_controller(genome_path, args.config, render=args.render)
    else:
        was_interrupted, genome_path = run_evolution(args.config)

        if was_interrupted:
            print("\nEvolution interrupted. Testing best controller found...")

        if genome_path and genome_path.exists():
            test_best_controller(genome_path, args.config, render=args.render)


if __name__ == "__main__":
    main()
