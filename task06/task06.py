import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plotting import plot_convergence

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Load data
DATA_PATH = Path(__file__).parent.parent / "data" / "data"


def load_data() -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Load classification data from file.
    Returns:
        (features, labels) where features is list of (x, y) tuples
    """
    features = []
    labels = []

    with open(DATA_PATH) as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            features.append((x, y))
            labels.append(label)

    return features, labels


def activation_function(x: float) -> float:
    """
    Activation function: φ(x) = 2/(1 + exp(-2x)) - 1
    """
    return 2.0 / (1.0 + math.exp(-2.0 * x)) - 1.0


def predict(weights: Tuple[float, float, float], x: float, y: float) -> int:
    """
    Predict class using the ANN.

    Args:
        weights: (w0, w1, w2) - bias weight, x weight, y weight
        x, y: Input features

    Returns:
        0 if φ < 0, else 1
    """
    w0, w1, w2 = weights
    z = w0 * 1.0 + w1 * x + w2 * y  # 1.0 is the bias input
    output = activation_function(z)
    return 0 if output < 0 else 1


def fitness(
    weights: Tuple[float, float, float],
    features: List[Tuple[float, float]],
    labels: List[int],
) -> float:
    """
    Calculate fitness as fraction of correctly classified examples.
    """
    correct = 0
    for (x, y), true_label in zip(features, labels):
        predicted_label = predict(weights, x, y)
        if predicted_label == true_label:
            correct += 1
    return correct / len(labels)


def random_weights() -> Tuple[float, float, float]:
    """Generate random initial weights in [-1, 1]."""
    return (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))


def mutate(
    weights: Tuple[float, float, float],
    mutation_std: float = 0.5,
) -> Tuple[float, float, float]:
    """Mutate weights by adding Gaussian noise."""
    w0, w1, w2 = weights
    return (
        w0 + random.gauss(0, mutation_std),
        w1 + random.gauss(0, mutation_std),
        w2 + random.gauss(0, mutation_std),
    )


def tournament_select(
    population: List[Tuple[float, float, float]],
    fitness_values: List[float],
) -> Tuple[float, float, float]:
    """Select parent via size-2 tournament."""
    idx1, idx2 = random.sample(range(len(population)), 2)
    return (
        population[idx1]
        if fitness_values[idx1] >= fitness_values[idx2]
        else population[idx2]
    )


def evolve(
    features: List[Tuple[float, float]],
    labels: List[int],
    population_size: int = 50,
    generations: int = 200,
    mutation_std: float = 0.5,
    print_every: int = 20,
) -> Tuple[Tuple[float, float, float], float, List[float], List[float]]:
    """
    Evolve an ANN classifier.

    Returns:
        (best_weights, best_fitness, best_history, avg_history)
    """
    population = [random_weights() for _ in range(population_size)]
    best_weights = population[0]
    best_fit = fitness(best_weights, features, labels)

    best_history: List[float] = []
    avg_history: List[float] = []

    print(f"{'Gen':>5} | {'Best Fitness':>14} | {'Avg Fitness':>14}")
    print("-" * 50)

    for generation in range(generations):
        fitness_values = [fitness(weights, features, labels) for weights in population]
        avg_fit = sum(fitness_values) / len(fitness_values)
        max_idx = fitness_values.index(max(fitness_values))

        if fitness_values[max_idx] > best_fit:
            best_fit = fitness_values[max_idx]
            best_weights = population[max_idx]

        best_history.append(best_fit)
        avg_history.append(avg_fit)

        if generation % print_every == 0 or generation == generations - 1:
            print(f"{generation:5d} | {best_fit:14.10f} | {avg_fit:14.10f}")

        # Create new population
        new_population: List[Tuple[float, float, float]] = [best_weights]

        while len(new_population) < population_size:
            parent = tournament_select(population, fitness_values)
            child = mutate(parent, mutation_std)
            new_population.append(child)

        population = new_population

    return best_weights, best_fit, best_history, avg_history


def plot_decision_boundary(
    weights: Tuple[float, float, float],
    features: List[Tuple[float, float]],
    labels: List[int],
    output_path: str = "plots/decision_boundary.png",
) -> None:
    """
    Plot the data points and the decision boundary line.

    The decision boundary is: y = -w0/w2 - (w1/w2)*x
    """
    w0, w1, w2 = weights

    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate data by class
    class_0_x = [x for (x, y), label in zip(features, labels) if label == 0]
    class_0_y = [y for (x, y), label in zip(features, labels) if label == 0]
    class_1_x = [x for (x, y), label in zip(features, labels) if label == 1]
    class_1_y = [y for (x, y), label in zip(features, labels) if label == 1]

    # Plot data points
    ax.scatter(class_0_x, class_0_y, c="blue", label="Class 0", s=100, alpha=0.6)
    ax.scatter(class_1_x, class_1_y, c="red", label="Class 1", s=100, alpha=0.6)

    # Plot decision boundary
    if w2 != 0:
        x_min = min(x for x, y in features) - 0.2
        x_max = max(x for x, y in features) + 0.2
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -w0 / w2 - (w1 / w2) * x_line
        ax.plot(x_line, y_line, "k--", linewidth=2, label="Decision Boundary")

    ax.set_xlabel("Feature X")
    ax.set_ylabel("Feature Y")
    ax.set_title("ANN Classifier Decision Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved decision boundary plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    plots_dir = CONFIG["output"]["plots_dir"]
    Path(plots_dir).mkdir(exist_ok=True)

    print("=" * 80)
    print("Task 6: Optimal Classification with Evolutionary Algorithms")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    features, labels = load_data()
    print(f"Loaded {len(features)} examples")
    print(f"Class 0: {sum(1 for label in labels if label == 0)} examples")
    print(f"Class 1: {sum(1 for label in labels if label == 1)} examples")

    # Evolve classifier
    print("\n" + "=" * 80)
    print("Evolving ANN classifier...")
    print("=" * 80)
    best_weights, best_fitness, best_hist, avg_hist = evolve(
        features, labels, population_size=50, generations=200
    )

    print("\n" + "=" * 80)
    print("Best ANN found:")
    print(
        f"  Weights: w0={best_weights[0]:.6f}, w1={best_weights[1]:.6f}, w2={best_weights[2]:.6f}"
    )
    print(
        f"  Fitness: {best_fitness:.6f} ({int(best_fitness * len(labels))}/{len(labels)} correct)"
    )

    # Plot convergence
    plot_convergence(
        best_hist,
        avg_hist,
        output_path=f"{plots_dir}/classification_convergence.png",
    )

    # Plot decision boundary
    plot_decision_boundary(
        best_weights,
        features,
        labels,
        output_path=f"{plots_dir}/decision_boundary.png",
    )

    print("\n" + "=" * 80)
    print("All plots saved to plots/ directory")
    print("=" * 80)
