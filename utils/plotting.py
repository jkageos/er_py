from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_convergence",
    "plot_parameter_comparison",
]


def plot_ackley_3d(
    ackley_func,
    output_path: str = "plots/ackley_3d_surface.png",
    resolution: int = 50,
    bounds: Tuple[float, float] = (-32.768, 32.768),
) -> None:
    """
    Plot the 3D Ackley function surface (x-y plane with z as fitness).

    Args:
        ackley_func: The Ackley function to plot
        output_path: Where to save the plot
        resolution: Grid resolution for the surface
        bounds: (min, max) bounds for x and y axes
    """
    x_min, x_max = bounds
    y_min, y_max = bounds

    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate Ackley function at z=0
    Z = np.array(
        [
            [ackley_func(X[i, j], Y[i, j], 0) for j in range(resolution)]
            for i in range(resolution)
        ]
    )

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(x, y, 0)")
    ax.set_title("Ackley Function (z=0 plane)")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved 3D surface plot to {output_path}")
    plt.close()


def plot_ackley_2d_contour(
    ackley_func,
    output_path: str = "plots/ackley_2d_contour.png",
    resolution: int = 100,
    bounds: Tuple[float, float] = (-32.768, 32.768),
) -> None:
    """
    Plot a 2D contour map of the Ackley function (x-y plane, z=0).

    Args:
        ackley_func: The Ackley function to plot
        output_path: Where to save the plot
        resolution: Grid resolution for the contour
        bounds: (min, max) bounds for x and y axes
    """
    x_min, x_max = bounds
    y_min, y_max = bounds

    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate Ackley function at z=0
    Z = np.array(
        [
            [ackley_func(X[i, j], Y[i, j], 0) for j in range(resolution)]
            for i in range(resolution)
        ]
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot contour
    levels = np.linspace(Z.min(), Z.max(), 20)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    ax.contour(X, Y, Z, levels=levels, colors="black", alpha=0.3, linewidths=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Ackley Function Contour Map (z=0 plane)")
    ax.set_aspect("equal")

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("f(x, y, 0)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved 2D contour plot to {output_path}")
    plt.close()


def plot_convergence(
    best_history: List[float],
    avg_history: List[float],
    output_path: str = "plots/convergence.png",
) -> None:
    """
    Plot convergence of best and average fitness over generations.

    Args:
        best_history: Best fitness per generation
        avg_history: Average fitness per generation
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = np.arange(len(best_history))
    ax.plot(generations, best_history, linewidth=2, color="blue", label="Best Fitness")
    ax.plot(generations, avg_history, linewidth=2, color="orange", label="Avg Fitness")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolutionary Algorithm Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved convergence plot to {output_path}")
    plt.close()


def plot_solution_on_contour(
    ackley_func,
    solutions: List[Tuple[float, float, float]],
    output_path: str = "plots/solution_on_contour.png",
    resolution: int = 100,
    bounds: Tuple[float, float] = (-32.768, 32.768),
) -> None:
    """
    Plot the discovered solutions on top of the Ackley contour map.

    Args:
        ackley_func: The Ackley function to plot
        solutions: List of (x, y, z) solutions
        output_path: Where to save the plot
        resolution: Grid resolution for the contour
        bounds: (min, max) bounds for x and y axes
    """
    x_min, x_max = bounds
    y_min, y_max = bounds

    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate Ackley function at z=0
    Z = np.array(
        [
            [ackley_func(X[i, j], Y[i, j], 0) for j in range(resolution)]
            for i in range(resolution)
        ]
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot contour
    levels = np.linspace(Z.min(), Z.max(), 20)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    ax.contour(X, Y, Z, levels=levels, colors="black", alpha=0.3, linewidths=0.5)

    # Plot solutions (only x-y projection)
    solution_x = [sol[0] for sol in solutions]
    solution_y = [sol[1] for sol in solutions]

    ax.scatter(
        solution_x,
        solution_y,
        c="red",
        s=100,
        marker="*",
        edgecolors="darkred",
        linewidths=2,
        label="Solutions",
        zorder=5,
    )

    # Mark global optimum at (0, 0, 0)
    ax.scatter(
        0,
        0,
        c="yellow",
        s=200,
        marker="x",
        linewidths=3,
        label="Global Optimum (0, 0)",
        zorder=6,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Solutions Found on Ackley Contour Map (z=0 plane)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("f(x, y, 0)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved solution contour plot to {output_path}")
    plt.close()


def plot_statistics(
    fitness_values: List[float],
    output_path: str = "plots/statistics.png",
) -> None:
    """
    Plot histogram and statistics of fitness values from multiple runs.

    Args:
        fitness_values: List of final fitness values from multiple runs
        output_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(fitness_values, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.axvline(
        np.mean(fitness_values),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(fitness_values):.4f}",
    )
    ax1.axvline(
        np.median(fitness_values),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(fitness_values):.4f}",
    )
    ax1.set_xlabel("Fitness")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Best Fitness Values")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(fitness_values, vert=True)
    ax2.set_ylabel("Fitness")
    ax2.set_title("Box Plot of Fitness Values")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved statistics plot to {output_path}")
    plt.close()


def plot_convergence_multiple_runs(
    fitness_histories: List[List[float]],
    output_path: str = "plots/convergence_multiple_runs.png",
    labels: Optional[List[str]] = None,
) -> None:
    """
    Plot convergence curves for multiple runs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, history in enumerate(fitness_histories):
        label = labels[i] if labels else f"Run {i + 1}"
        generations = np.arange(len(history))
        ax.plot(generations, history, linewidth=1.5, alpha=0.7, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Convergence Across Runs")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved multiple convergence plot to {output_path}")
    plt.close()


def plot_parameter_comparison(
    results_dict: dict,
    output_path: str = "plots/parameter_comparison.png",
) -> None:
    """
    Compare results across different parameter settings.

    Args:
        results_dict: Dict with parameter names as keys and lists of fitness values as values
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = list(results_dict.keys())
    data = [results_dict[label] for label in labels]

    bp = ax.boxplot(data, patch_artist=True)  # removed labels kwarg
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.set_ylabel("Best Fitness")
    ax.set_title("Performance Comparison Across Parameter Settings")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved parameter comparison plot to {output_path}")
    plt.close()


def plot_population_heatmap(
    population_samples: List[Tuple[float, float, float]],
    output_path: str = "plots/population_heatmap.png",
    resolution: int = 100,
    bounds: Tuple[float, float] = (-32.768, 32.768),
) -> None:
    """
    Create a heatmap showing which regions the population explored.

    Args:
        ackley_func: The Ackley function
        population_samples: List of population positions sampled over time
        output_path: Where to save the plot
        resolution: Grid resolution
        bounds: Search space bounds
    """
    x_min, x_max = bounds
    y_min, y_max = bounds

    # Create grid for visited counts
    visited_grid = np.zeros((resolution, resolution))

    # Map positions to grid
    for x, y, z in population_samples:
        grid_x = int((x - x_min) / (x_max - x_min) * (resolution - 1))
        grid_y = int((y - y_min) / (y_max - y_min) * (resolution - 1))

        # Clip to valid range
        grid_x = max(0, min(resolution - 1, grid_x))
        grid_y = max(0, min(resolution - 1, grid_y))

        visited_grid[grid_y, grid_x] += 1

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(
        visited_grid,
        extent=(x_min, x_max, y_min, y_max),  # tuple to satisfy imshow typing
        origin="lower",
        cmap="hot",
        aspect="auto",
    )

    # Mark global optimum
    ax.scatter(
        0,
        0,
        c="cyan",
        s=200,
        marker="x",
        linewidths=3,
        label="Global Optimum",
        zorder=5,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Population Search Space Coverage Heatmap")
    ax.legend()

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Visits")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved population heatmap to {output_path}")
    plt.close()


def plot_fitness_vs_parameter(
    parameter_values: List[float],
    fitness_values: List[float],
    parameter_name: str,
    output_path: str = "plots/fitness_vs_parameter.png",
) -> None:
    """
    Plot fitness as a function of a parameter.

    Args:
        parameter_values: List of parameter values tested
        fitness_values: List of average fitness values
        parameter_name: Name of the parameter
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        parameter_values,
        fitness_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="blue",
    )

    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Average Best Fitness")
    ax.set_title(f"Optimization Performance vs {parameter_name}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved fitness vs parameter plot to {output_path}")
    plt.close()
