from typing import List, Optional, Tuple

import numpy as np

from task07.ann import ANN
from task07.fitness import evaluate_fitness


class EvolutionaryAlgorithm:
    """Evolutionary algorithm to evolve ANN for XOR problem."""

    def __init__(
        self,
        population_size: int = 1000,
        mutation_rate: float = 0.3,
        mutation_std: float = 1.0,
        elite_size: int = 50,
        adaptive_mutation: bool = True,
    ):
        """
        Initialize evolutionary algorithm.

        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutating each gene
            mutation_std: Standard deviation of mutation
            elite_size: Number of best individuals to preserve
            adaptive_mutation: Whether to use adaptive mutation
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elite_size = elite_size
        self.adaptive_mutation = adaptive_mutation

        self.population: List[ANN] = []
        self.fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.best_individual: Optional[ANN] = None
        self.best_fitness: float = 0.0
        self.generations_without_improvement = 0

    def initialize_population(self) -> None:
        """Initialize population with random ANNs with diverse initialization."""
        self.population = []
        for _ in range(self.population_size):
            # Use larger initialization range for more diversity
            weights = np.random.uniform(-3, 3, 6)
            biases = np.random.uniform(-3, 3, 3)
            self.population.append(ANN(weights, biases))

    def evaluate_population(self) -> Tuple[List[float], float, float]:
        """
        Evaluate fitness of entire population.

        Returns:
            Tuple of (fitness_list, best_fitness, avg_fitness)
        """
        fitness_list = [evaluate_fitness(ann) for ann in self.population]
        best_fitness = max(fitness_list)
        avg_fitness = float(np.mean(fitness_list))

        return fitness_list, best_fitness, avg_fitness

    def selection(self, fitness_list: List[float]) -> List[ANN]:
        """
        Tournament selection with adaptive tournament size.

        Args:
            fitness_list: Fitness values for population

        Returns:
            Selected parents
        """
        selected = []
        # Increase tournament size when stuck
        tournament_size = 5 + min(self.generations_without_improvement // 100, 5)

        for _ in range(self.population_size):
            # Random tournament
            indices = np.random.choice(len(self.population), tournament_size)
            tournament_fitness = [fitness_list[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx])

        return selected

    def crossover(self, parent1: ANN, parent2: ANN) -> ANN:
        """
        Uniform crossover instead of single-point.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Child ANN
        """
        genes1 = parent1.get_genes()
        genes2 = parent2.get_genes()

        # Uniform crossover: randomly choose from each parent
        child_genes = np.where(np.random.random(len(genes1)) < 0.5, genes1, genes2)

        child = ANN()
        child.set_genes(child_genes)
        return child

    def mutate(self, ann: ANN) -> None:
        """
        Mutate ANN genes with adaptive mutation strength.

        Args:
            ann: ANN to mutate
        """
        genes = ann.get_genes()

        # Adaptive mutation: increase mutation when stuck
        if self.adaptive_mutation and self.generations_without_improvement > 200:
            current_mutation_rate = min(0.5, self.mutation_rate * 1.5)
            current_mutation_std = min(2.0, self.mutation_std * 1.5)
        else:
            current_mutation_rate = self.mutation_rate
            current_mutation_std = self.mutation_std

        for i in range(len(genes)):
            if np.random.random() < current_mutation_rate:
                genes[i] += np.random.normal(0, current_mutation_std)

        ann.set_genes(genes)

    def evolve(self, max_generations: int = 2000, target_fitness: float = 0.99) -> bool:
        """
        Run the evolutionary algorithm.

        Args:
            max_generations: Maximum number of generations
            target_fitness: Target fitness to stop evolution

        Returns:
            True if target fitness reached, False otherwise
        """
        self.initialize_population()
        last_best_fitness = 0.0

        for generation in range(max_generations):
            # Evaluate population
            fitness_list, best_fitness, avg_fitness = self.evaluate_population()

            # Track history
            self.fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Track stagnation
            if best_fitness > last_best_fitness + 1e-6:
                self.generations_without_improvement = 0
                last_best_fitness = best_fitness
            else:
                self.generations_without_improvement += 1

            # Update best individual
            best_idx = np.argmax(fitness_list)
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_individual = ANN(
                    self.population[best_idx].weights.copy(),
                    self.population[best_idx].biases.copy(),
                )

            # Print progress
            if (generation + 1) % 100 == 0:
                print(
                    f"Generation {generation + 1}: Best = {best_fitness:.6f}, "
                    f"Avg = {avg_fitness:.6f}, Stagnant = {self.generations_without_improvement}"
                )

            # Check if target reached
            if best_fitness >= target_fitness:
                print(f"Target fitness reached at generation {generation + 1}")
                return True

            # Diversity injection when stuck
            if self.generations_without_improvement > 500:
                print(f"  Injecting diversity at generation {generation + 1}")
                # Replace worst 20% with random individuals
                num_random = self.population_size // 5
                sorted_indices = np.argsort(fitness_list)
                for idx in sorted_indices[:num_random]:
                    self.population[idx] = ANN()
                self.generations_without_improvement = 0

            # Preserve elite
            elite_indices = np.argsort(fitness_list)[-self.elite_size :]
            elite = [self.population[i] for i in elite_indices]

            # Selection and reproduction
            selected = self.selection(fitness_list)
            new_population = elite.copy()

            while len(new_population) < self.population_size:
                parent1 = selected[np.random.randint(len(selected))]
                parent2 = selected[np.random.randint(len(selected))]

                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population[: self.population_size]

        print(
            f"Maximum generations ({max_generations}) reached without target fitness."
        )
        return False
