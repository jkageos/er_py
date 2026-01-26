"""Task 9: Evolve a controller for the ring problem."""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import yaml

from task09.evolutionary_algorithm import RingEvolutionaryAlgorithm
from task09.ring_environment import RingAgent, RingEnvironment
from utils.plotting import plot_convergence

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Extract ring parameters from config
CELLS_PER_HALF = CONFIG["ring"]["cells_per_half"]
POPULATION_SIZE = CONFIG["ring"]["ea"]["population_size"]
MAX_GENERATIONS = CONFIG["ring"]["ea"]["max_generations"]
TARGET_FITNESS = CONFIG["ring"]["ea"]["target_fitness"]
MUTATION_RATE = CONFIG["ring"]["ea"]["mutation_rate"]
MUTATION_STD = CONFIG["ring"]["ea"]["mutation_std"]
ELITE_SIZE = CONFIG["ring"]["ea"]["elite_size"]
NUM_TRIALS = CONFIG["ring"]["fitness"]["num_trials"]
STEPS_PER_TRIAL = CONFIG["ring"]["fitness"]["steps_per_trial"]
THRESHOLD = CONFIG["ring"]["fitness"]["threshold"]
PRINT_EVERY = CONFIG["ring"]["output"]["print_every"]
TEST_START_POSITIONS = CONFIG["ring"]["output"]["test_start_positions"]
TRAJECTORY_STEPS = CONFIG["ring"]["output"]["trajectory_steps"]
PLOTS_DIR = CONFIG["output"]["plots_dir"]
PARALLEL_ENABLED = CONFIG["parallel"]["enabled"]
N_WORKERS = CONFIG["parallel"]["n_workers"]


def visualize_controller_decisions(
    weights: list[float], environment: RingEnvironment, threshold: float = 0.0
) -> None:
    """
    Visualize the decisions made by the controller for each cell.

    Args:
        weights: Neural network weights
        environment: Ring environment
        threshold: Action threshold
    """
    agent = RingAgent(weights, threshold)

    print("\nController Decisions for Each Cell:")
    print("=" * 80)
    print(
        f"{'Position':>8} | {'Half':>10} | {'Cell Number':>12} | {'Weight':>10} | {'Action':>6}"
    )
    print("-" * 80)

    for position in range(environment.total_cells):
        cell_number = environment.get_cell_number(position)
        half = "LEFT" if environment.is_left_half(position) else "RIGHT"
        action = agent.decide_action(cell_number)
        weight = weights[cell_number]

        print(
            f"{position:8d} | {half:>10} | {cell_number:12d} | {weight:10.4f} | {action:>6}"
        )


def plot_ring_decisions(
    weights: list[float],
    environment: RingEnvironment,
    threshold: float = 0.0,
    output_path: str = "plots/ring_decisions.png",
) -> None:
    """
    Plot a circular diagram showing the ring with motion directions.

    Args:
        weights: Neural network weights
        environment: Ring environment
        threshold: Action threshold
        output_path: Where to save the plot
    """
    agent = RingAgent(weights, threshold)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")

    # Ring parameters
    outer_radius = 10
    inner_radius = 7
    arrow_radius = outer_radius + 1.2  # Place arrows outside the ring
    num_cells = environment.total_cells

    # Calculate arc angle per cell
    arc_angle = 2 * np.pi / num_cells  # angle per cell in radians
    arc_length = arrow_radius * arc_angle  # arc length = radius * angle

    # Draw ring cells
    for position in range(num_cells):
        # Calculate angles for this cell
        angle_start = position * 2 * np.pi / num_cells
        angle_end = (position + 1) * 2 * np.pi / num_cells
        angle_mid = (angle_start + angle_end) / 2

        # Determine color based on half
        if environment.is_left_half(position):
            color = "lightgreen"
            edge_color = "darkgreen"
        else:
            color = "lightcoral"
            edge_color = "darkred"

        # Draw cell wedge
        theta = np.linspace(angle_start, angle_end, 20)
        x_outer = outer_radius * np.cos(theta)
        y_outer = outer_radius * np.sin(theta)
        x_inner = inner_radius * np.cos(theta)
        y_inner = inner_radius * np.sin(theta)

        # Create wedge
        x_wedge = np.concatenate([x_outer, x_inner[::-1]])
        y_wedge = np.concatenate([y_outer, y_inner[::-1]])
        ax.fill(x_wedge, y_wedge, color=color, edgecolor=edge_color, linewidth=1.5)

        # Get cell number and action
        cell_number = environment.get_cell_number(position)
        action = agent.decide_action(cell_number)

        # Position for text (middle of wedge)
        text_radius = (outer_radius + inner_radius) / 2
        text_x = text_radius * np.cos(angle_mid)
        text_y = text_radius * np.sin(angle_mid)

        # Display cell number
        ax.text(
            text_x,
            text_y,
            str(cell_number),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Draw arrow showing direction on outer ring
        # Arrow should span most of the cell's arc length (80%)
        arrow_length = arc_length * 0.8

        # Determine start position based on action direction
        if action == "CW":
            # Clockwise: start from the beginning of cell (angle_start + small offset)
            # and point toward the end
            arrow_start_angle = angle_start + arc_angle * 0.1  # 10% offset from start
            arrow_color = "blue"
            # Direction: tangent in CW direction (negative angle direction in standard coords)
            arrow_dx = -np.sin(arrow_start_angle) * arrow_length
            arrow_dy = np.cos(arrow_start_angle) * arrow_length
        else:  # CCW
            # Counterclockwise: start from the end of cell (angle_end - small offset)
            # and point toward the beginning
            arrow_start_angle = angle_end - arc_angle * 0.1  # 10% offset from end
            arrow_color = "purple"
            # Direction: tangent in CCW direction (positive angle direction in standard coords)
            arrow_dx = np.sin(arrow_start_angle) * arrow_length
            arrow_dy = -np.cos(arrow_start_angle) * arrow_length

        # Calculate arrow start position
        arrow_x = arrow_radius * np.cos(arrow_start_angle)
        arrow_y = arrow_radius * np.sin(arrow_start_angle)

        # Draw arrow
        ax.arrow(
            arrow_x,
            arrow_y,
            arrow_dx,
            arrow_dy,
            head_width=0.4,
            head_length=0.3,
            fc=arrow_color,
            ec=arrow_color,
            linewidth=2.5,
            length_includes_head=True,
        )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="darkgreen", label="Left Half (Goal)"),
        Patch(facecolor="lightcoral", edgecolor="darkred", label="Right Half"),
        Patch(facecolor="blue", label="Clockwise (CW)"),
        Patch(facecolor="purple", label="Counterclockwise (CCW)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # Add title with annotations
    ax.set_title(
        "Ring Controller Motion Directions\n"
        "Numbers show cell labels (0-19), Arrows show movement direction",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add boundary marker
    ax.plot(0, 0, "k*", markersize=15, label="Center")
    ax.text(
        0,
        -arrow_radius - 1.5,
        "← Left Half | Right Half →",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    # Set axis limits and remove axes
    margin = 2
    ax.set_xlim(-arrow_radius - margin, arrow_radius + margin)
    ax.set_ylim(-arrow_radius - margin, arrow_radius + margin)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["output"]["dpi"], bbox_inches="tight")
    print(f"Saved ring decisions diagram to {output_path}")
    plt.close()


def plot_ring_trajectory(
    weights: list[float],
    environment: RingEnvironment,
    steps: int,
    threshold: float = 0.0,
    output_path: str = "plots/ring_trajectory.png",
) -> None:
    """
    Plot a trajectory of the agent on the ring.

    Args:
        weights: Neural network weights
        environment: Ring environment
        steps: Number of steps to simulate
        threshold: Action threshold
        output_path: Where to save the plot
    """
    agent = RingAgent(weights, threshold)

    # Start from right half
    start_pos = environment.total_cells - 1
    _, trajectory = agent.simulate(environment, steps, start_pos)

    # Mark left/right halves
    left_half = []
    right_half = []

    for step, pos in enumerate(trajectory):
        if environment.is_left_half(pos):
            left_half.append(step)
        else:
            right_half.append(step)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot trajectory as position over time
    ax1.plot(range(len(trajectory)), trajectory, linewidth=1.5, color="blue")
    ax1.axhline(
        y=environment.cells_per_half - 0.5,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Left/Right Boundary",
    )
    ax1.fill_between(
        range(len(trajectory)),
        0,
        environment.cells_per_half,
        alpha=0.2,
        color="green",
        label="Left Half (Goal)",
    )
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Position on Ring")
    ax1.set_title("Agent Trajectory on Ring")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot time spent in each half
    time_in_left = len(left_half)
    time_in_right = len(right_half)

    ax2.bar(
        ["Left Half (Goal)", "Right Half"],
        [time_in_left, time_in_right],
        color=["green", "red"],
        alpha=0.7,
    )
    ax2.set_ylabel("Time Steps")
    ax2.set_title(
        f"Time Distribution (Left: {time_in_left}/{steps}, {100 * time_in_left / steps:.1f}%)"
    )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["output"]["dpi"])
    print(f"\nSaved trajectory plot to {output_path}")
    plt.close()


def main():
    """Main function to run ring problem evolution."""
    print("Starting Ring Problem Evolution")
    print("=" * 80)
    print(f"\nConfiguration from {CONFIG_PATH}:")
    print(f"  Cells per half: {CELLS_PER_HALF}")
    print(f"  Population size: {POPULATION_SIZE}")
    print(f"  Max generations: {MAX_GENERATIONS}")
    print(f"  Target fitness: {TARGET_FITNESS}")
    print(f"  Mutation rate: {MUTATION_RATE}")
    print(f"  Mutation std: {MUTATION_STD}")
    print(f"  Elite size: {ELITE_SIZE}")
    print(f"  Trials per evaluation: {NUM_TRIALS}")
    print(f"  Steps per trial: {STEPS_PER_TRIAL}")
    print(f"  Parallel processing: {PARALLEL_ENABLED}")
    if PARALLEL_ENABLED:
        from utils.parallel import get_optimal_workers

        workers = get_optimal_workers(N_WORKERS)
        print(f"  Worker processes: {workers}")

    # Create plots directory
    Path(PLOTS_DIR).mkdir(exist_ok=True)

    # Create environment
    environment = RingEnvironment(cells_per_half=CELLS_PER_HALF)

    print(
        f"\nEnvironment created with {environment.total_cells} cells "
        f"({environment.cells_per_half} per half)"
    )

    # Create evolutionary algorithm
    ea = RingEvolutionaryAlgorithm(
        environment=environment,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        mutation_std=MUTATION_STD,
        elite_size=ELITE_SIZE,
        num_trials=NUM_TRIALS,
        steps_per_trial=STEPS_PER_TRIAL,
        threshold=THRESHOLD,
        print_every=PRINT_EVERY,
        parallel=PARALLEL_ENABLED,
        n_workers=N_WORKERS,
    )

    # Evolve controller
    print("\nEvolving controller...")
    print("=" * 80)

    import time

    start_time = time.time()
    ea.evolve(max_generations=MAX_GENERATIONS, target_fitness=TARGET_FITNESS)
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print(f"Best Fitness: {ea.best_fitness:.6f}")
    print(f"Generations: {len(ea.fitness_history)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

    # Visualize best controller
    if ea.best_individual:
        visualize_controller_decisions(
            ea.best_individual, environment, threshold=ea.threshold
        )

        # Plot ring decisions diagram
        plot_ring_decisions(
            ea.best_individual,
            environment,
            threshold=ea.threshold,
            output_path=f"{PLOTS_DIR}/ring_decisions.png",
        )

        # Plot convergence
        plot_convergence(
            ea.fitness_history,
            ea.avg_fitness_history,
            output_path=f"{PLOTS_DIR}/ring_convergence.png",
        )
        print(f"\nSaved convergence plot to {PLOTS_DIR}/ring_convergence.png")

        # Plot trajectory
        plot_ring_trajectory(
            ea.best_individual,
            environment,
            steps=TRAJECTORY_STEPS,
            threshold=ea.threshold,
            output_path=f"{PLOTS_DIR}/ring_trajectory.png",
        )

        # Test with multiple starting positions
        print("\n" + "=" * 80)
        print("Testing with different starting positions:")
        print("=" * 80)

        agent = RingAgent(ea.best_individual, ea.threshold)
        for start_pos in TEST_START_POSITIONS:
            time_in_left, _ = agent.simulate(environment, 100, start_pos)
            print(
                f"Start position {start_pos:2d}: {time_in_left:3d}/100 steps in left half "
                f"({100 * time_in_left / 100:.1f}%)"
            )

    else:
        print("\nNo solution found!")


if __name__ == "__main__":
    main()
