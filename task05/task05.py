import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plotting import (
    plot_convergence,
    plot_parameter_comparison,
)

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Extract bounds from config
X_MIN = CONFIG["ackley"]["x_min"]
X_MAX = CONFIG["ackley"]["x_max"]
Y_MIN = CONFIG["ackley"]["y_min"]
Y_MAX = CONFIG["ackley"]["y_max"]
Z_MIN = CONFIG["ackley"]["z_min"]
Z_MAX = CONFIG["ackley"]["z_max"]

# Extract EA parameters from config
DEFAULT_POPULATION_SIZE = CONFIG["ea"]["population_size"]
DEFAULT_GENERATIONS = CONFIG["ea"]["generations"]
DEFAULT_MUTATION_RATE = CONFIG["ea"]["mutation_rate"]
DEFAULT_ELITISM = CONFIG["ea"]["elitism"]
GAUSSIAN_STD = CONFIG["ea"]["gaussian_std"]

# Extract output parameters from config
PRINT_EVERY = CONFIG["output"]["print_every"]


def ackley(x: float, y: float, z: float) -> float:
    """
    Calculate the Ackley function value.

    f(x, y, z) = -20 * exp(-0.2 * sqrt(1/3 * (x^2 + y^2 + z^2)))
                 - exp(1/3 * (cos(2π*x) + cos(2π*y) + cos(2π*z)))
                 + 20 + exp(1)
    """
    sum_squares = (x**2 + y**2 + z**2) / 3
    sum_cos = (
        math.cos(2 * math.pi * x)
        + math.cos(2 * math.pi * y)
        + math.cos(2 * math.pi * z)
    ) / 3

    term1 = -20 * math.exp(-0.2 * math.sqrt(sum_squares))
    term2 = -math.exp(sum_cos)
    term3 = 20 + math.exp(1)

    return term1 + term2 + term3


def fitness(x: float, y: float, z: float) -> float:
    """Convert minimization problem to maximization: fitness = 1 / (f(x,y,z) + 1)"""
    return 1 / (ackley(x, y, z) + 1)


def random_individual() -> Tuple[float, float, float]:
    """Generate a random individual within bounds."""
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    z = random.uniform(Z_MIN, Z_MAX)
    return (x, y, z)


def mutate(
    individual: Tuple[float, float, float], mutation_rate: float = DEFAULT_MUTATION_RATE
) -> Tuple[float, float, float]:
    """
    Mutate an individual by adding Gaussian noise to each dimension
    with a certain probability.
    """
    x, y, z = individual

    if random.random() < mutation_rate:
        x = x + random.gauss(0, GAUSSIAN_STD)
        x = max(X_MIN, min(X_MAX, x))

    if random.random() < mutation_rate:
        y = y + random.gauss(0, GAUSSIAN_STD)
        y = max(Y_MIN, min(Y_MAX, y))

    if random.random() < mutation_rate:
        z = z + random.gauss(0, GAUSSIAN_STD)
        z = max(Z_MIN, min(Z_MAX, z))

    return (x, y, z)


def tournament_select(
    population: List[Tuple[float, float, float]], fitness_values: List[float]
) -> Tuple[float, float, float]:
    """Select parent via size-2 tournament."""
    idx1, idx2 = random.sample(range(len(population)), 2)
    return (
        population[idx1]
        if fitness_values[idx1] >= fitness_values[idx2]
        else population[idx2]
    )


def evolve(
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    print_every: int = PRINT_EVERY,
    elitism: bool = DEFAULT_ELITISM,
) -> Tuple[Tuple[float, float, float], float, List[float], List[float]]:
    """
    Run the evolutionary algorithm to solve the Ackley problem.

    Returns:
        (best_individual, best_fitness, best_history, avg_history)
    """
    population = [random_individual() for _ in range(population_size)]
    best_individual = population[0]
    best_fitness = fitness(*best_individual)

    best_history: List[float] = []
    avg_history: List[float] = []

    print(
        f"{'Gen':>5} | {'Best Fitness':>14} | {'Avg Fitness':>14} | {'Best Position':>40}"
    )
    print("-" * 80)

    for generation in range(generations):
        fitness_values = [fitness(ind[0], ind[1], ind[2]) for ind in population]
        avg_fit = sum(fitness_values) / len(fitness_values)
        max_idx = fitness_values.index(max(fitness_values))

        if fitness_values[max_idx] > best_fitness:
            best_fitness = fitness_values[max_idx]
            best_individual = population[max_idx]

        best_history.append(best_fitness)
        avg_history.append(avg_fit)

        if generation % print_every == 0 or generation == generations - 1:
            print(
                f"{generation:5d} | {best_fitness:14.10f} | {avg_fit:14.10f} | "
                f"({best_individual[0]:8.4f}, {best_individual[1]:8.4f}, {best_individual[2]:8.4f})"
            )

        new_population: List[Tuple[float, float, float]] = []
        if elitism and best_individual is not None:
            new_population.append(best_individual)

        while len(new_population) < population_size:
            parent = tournament_select(population, fitness_values)
            child = mutate(parent, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_individual, best_fitness, best_history, avg_history


def run_experiments(
    n_runs: int = 10,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    elitism: bool = DEFAULT_ELITISM,
) -> Tuple[List[float], List[Tuple[float, float, float]], List[List[float]]]:
    """Run multiple evolutionary algorithm runs and report statistics."""
    print(f"\nRunning {n_runs} experiments with:")
    print(f"  Population size: {population_size}")
    print(f"  Generations: {generations}")
    print(f"  Mutation rate: {mutation_rate}")
    print("=" * 80)

    best_fitnesses: List[float] = []
    best_positions: List[Tuple[float, float, float]] = []
    histories: List[List[float]] = []

    for run in range(n_runs):
        best_ind, best_fit, best_hist, _ = evolve(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            print_every=generations,
            elitism=elitism,
        )
        best_fitnesses.append(best_fit)
        best_positions.append(best_ind)
        histories.append(best_hist)
        print(
            f"Run {run + 1:2d}: fitness = {best_fit:.10f}, "
            f"position = ({best_ind[0]:8.4f}, {best_ind[1]:8.4f}, {best_ind[2]:8.4f})"
        )

    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print(f"  Average best fitness: {sum(best_fitnesses) / len(best_fitnesses):.10f}")
    print(f"  Best fitness: {max(best_fitnesses):.10f}")
    print(f"  Worst fitness: {min(best_fitnesses):.10f}")
    print(f"  Std dev: {np.std(best_fitnesses):.10f}")

    return best_fitnesses, best_positions, histories


def sweep_mutation() -> Dict[str, List[float]]:
    """Evaluate several mutation rates for comparison using config parameters."""
    mutation_rates = CONFIG["parameter_sweep"]["mutation_rates"]
    n_runs = CONFIG["parameter_sweep"]["n_runs_per_setting"]

    results: Dict[str, List[float]] = {}
    for mr in mutation_rates:
        fitnesses, _, _ = run_experiments(
            n_runs=n_runs,
            population_size=DEFAULT_POPULATION_SIZE,
            generations=DEFAULT_GENERATIONS,
            mutation_rate=mr,
        )
        results[f"m={mr}"] = fitnesses
    return results


if __name__ == "__main__":
    plots_dir = CONFIG["output"]["plots_dir"]
    Path(plots_dir).mkdir(exist_ok=True)

    print("=" * 80)
    print("Task 5: Classical Optimization - Ackley Problem")
    print("=" * 80)

    # Single run with detailed output
    print("\nSingle evolutionary run:\n")
    best_individual, best_fitness, best_hist, avg_hist = evolve()

    print("\n" + "=" * 80)
    print(f"Best solution found:")
    print(
        f"  Position: x={best_individual[0]:.6f}, y={best_individual[1]:.6f}, z={best_individual[2]:.6f}"
    )
    print(f"  Fitness: {best_fitness:.10f}")
    print(
        f"  Ackley value: {ackley(best_individual[0], best_individual[1], best_individual[2]):.10f}"
    )

    # Plot convergence (required: best and avg fitness per generation)
    plot_convergence(best_hist, avg_hist, output_path=f"{plots_dir}/convergence.png")

    # Multiple runs for statistical analysis
    print("\n" + "=" * 80)
    print("Multiple runs for statistical analysis:")
    print("=" * 80)
    best_fitnesses, best_positions, histories = run_experiments(n_runs=10)

    # Compare mutation rates
    mutation_results = sweep_mutation()
    plot_parameter_comparison(
        mutation_results, output_path=f"{plots_dir}/mutation_rate_comparison.png"
    )

    print("\n" + "=" * 80)
    print("All plots saved to plots/ directory")
    print("=" * 80)
