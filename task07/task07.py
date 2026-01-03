"""Task 7: Evolve an ANN to solve the XOR problem."""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from task07.ann import ANN
from task07.evolutionary_algorithm import EvolutionaryAlgorithm
from utils.plotting import plot_convergence


def main():
    """Main function to run XOR problem evolution."""
    print("Starting XOR Problem Evolution with ANN")
    print("=" * 50)

    # Parameters per task sheet: pop >= 1000, up to 2000 generations
    ea = EvolutionaryAlgorithm(
        population_size=1000,
        mutation_rate=0.3,
        mutation_std=1.0,
        elite_size=50,
        adaptive_mutation=True,
    )

    ea.evolve(max_generations=2000, target_fitness=0.99)

    print("\n" + "=" * 50)
    print("Evolution Complete!")
    print(f"Best Fitness: {ea.best_fitness:.6f}")
    print(f"Generations: {len(ea.fitness_history)}")

    # Test the best ANN on XOR cases
    if ea.best_individual is not None:
        print("\nTest Cases:")
        test_cases = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]
        for a, b, expected in test_cases:
            output = ea.best_individual.forward(a, b)
            mapped = (output + 1) / 2
            print(
                f"  {a} XOR {b} = {expected}, ANN output = {output:.4f} (mapped: {mapped:.4f})"
            )

        # Plot convergence
        plot_convergence(
            ea.fitness_history,
            ea.avg_fitness_history,
            output_path="plots/xor_convergence.png",
        )
        print("\nSaved convergence plot to plots/xor_convergence.png")

        # Plot ANN output over continuous input space
        plot_ann_output(ea.best_individual)
    else:
        print("\nNo solution found!")


def plot_ann_output(
    ann: ANN, output_path: str = "plots/xor_output_surface.png", resolution: int = 100
):
    """
    Plot the output of the ANN over the continuous input space [0, 1]².
    Uses raw ANN outputs in [-1, 1]; scatter z is cast to int(round(...)).
    """
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)

    # Mapped outputs in [0,1] for the surface (inverted to match reference)
    Z = np.zeros_like(X, dtype=float)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = 1.0 - ((ann.forward(float(X[i, j]), float(Y[i, j])) + 1) / 2)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Z, cmap="viridis", alpha=0.85, edgecolor="none", antialiased=True
    )

    ax.set_xlabel("Input a", fontsize=12)
    ax.set_ylabel("Input b", fontsize=12)
    ax.set_zlabel("ANN Output", fontsize=12)
    ax.set_title("XOR Output Over Continuous Input Space [0, 1]²", fontsize=13, pad=20)
    ax.set_zlim(0, 1)

    # Rotate view to align saddle from rear (0,0) to front (1,1)
    ax.view_init(elev=25, azim=-130)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label("Output Value", fontsize=11)

    # XOR corners, scatter z as int-rounded inverted mapped output
    test_points = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]
    for a, b, expected in test_points:
        mapped = 1.0 - ((ann.forward(float(a), float(b)) + 1) / 2)
        ax.scatter(
            float(a),
            float(b),
            int(round(mapped)),
            c="red",
            s=100,
            marker="o",
            edgecolors="darkred",
            linewidths=2,
            zorder=5,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved ANN output surface plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
