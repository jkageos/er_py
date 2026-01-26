"""Evolutionary algorithm for evolving ring controllers."""

import random
from typing import List, Optional, Tuple

import numpy as np

from task09.fitness import evaluate_fitness
from task09.ring_environment import RingEnvironment
from utils.parallel import parallel_map


class RingEvolutionaryAlgorithm:
    """Evolutionary algorithm for ring navigation problem."""

    def __init__(
        self,
        environment: RingEnvironment,
        population_size: int = 100,
        mutation_rate: float = 0.2,
        mutation_std: float = 0.5,
        elite_size: int = 10,
        num_trials: int = 10,
        steps_per_trial: int = 100,
        threshold: float = 0.0,
        print_every: int = 50,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ):
        """
        Initialize evolutionary algorithm.

        Args:
            environment: Ring environment
            population_size: Number of individuals in population
            mutation_rate: Probability of mutating each weight
            mutation_std: Standard deviation of mutation
            elite_size: Number of best individuals to preserve
            num_trials: Number of trials per fitness evaluation
            steps_per_trial: Steps per trial
            threshold: Action threshold
            print_every: Print progress every N generations
            parallel: Whether to use parallel processing
            n_workers: Number of worker processes (None for auto-detect)
        """
        self.environment = environment
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elite_size = elite_size
        self.num_trials = num_trials
        self.steps_per_trial = steps_per_trial
        self.threshold = threshold
        self.print_every = print_every
        self.parallel = parallel
        self.n_workers = n_workers

        self.population: List[List[float]] = []
        self.fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.best_individual: List[float] = []
        self.best_fitness: float = 0.0

    def initialize_population(self) -> None:
        """Initialize population with random weights."""
        self.population = []
        for _ in range(self.population_size):
            # Initialize weights in range [-2, 2] for diversity
            weights = [random.uniform(-2, 2) for _ in range(20)]
            self.population.append(weights)

    def _evaluate_single(self, weights: List[float]) -> float:
        """
        Evaluate a single individual's fitness.

        Args:
            weights: Neural network weights

        Returns:
            Fitness value
        """
        return evaluate_fitness(
            weights,
            self.environment,
            self.num_trials,
            self.steps_per_trial,
            self.threshold,
        )

    def evaluate_population(self) -> Tuple[List[float], float, float]:
        """
        Evaluate fitness of entire population.

        Returns:
            Tuple of (fitness_list, best_fitness, avg_fitness)
        """
        if self.parallel:
            # Parallel evaluation
            fitness_list = parallel_map(
                self._evaluate_single, self.population, n_workers=self.n_workers
            )
        else:
            # Sequential evaluation
            fitness_list = [
                self._evaluate_single(weights) for weights in self.population
            ]

        best_fitness = max(fitness_list)
        avg_fitness = float(np.mean(fitness_list))

        return fitness_list, best_fitness, avg_fitness

    def selection(self, fitness_list: List[float]) -> List[List[float]]:
        """
        Tournament selection.

        Args:
            fitness_list: Fitness values for population

        Returns:
            Selected parents
        """
        selected = []
        tournament_size = 3

        for _ in range(self.population_size):
            indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [fitness_list[i] for i in indices]
            winner_idx = indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(self.population[winner_idx])

        return selected

    def crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """
        Uniform crossover.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Child weights
        """
        child = []
        for w1, w2 in zip(parent1, parent2):
            child.append(w1 if random.random() < 0.5 else w2)
        return child

    def mutate(self, weights: List[float]) -> None:
        """
        Mutate weights in place.

        Args:
            weights: Weights to mutate
        """
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += random.gauss(0, self.mutation_std)

    def evolve(self, max_generations: int = 500, target_fitness: float = 0.95) -> bool:
        """
        Run the evolutionary algorithm.

        Args:
            max_generations: Maximum number of generations
            target_fitness: Target fitness to stop evolution

        Returns:
            True if target fitness reached, False otherwise
        """
        self.initialize_population()

        for generation in range(max_generations):
            # Evaluate population
            fitness_list, best_fitness, avg_fitness = self.evaluate_population()

            # Track history
            self.fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Update best individual
            best_idx = fitness_list.index(max(fitness_list))
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_individual = self.population[best_idx].copy()

            # Print progress
            if (generation + 1) % self.print_every == 0:
                print(
                    f"Generation {generation + 1}: Best = {best_fitness:.6f}, "
                    f"Avg = {avg_fitness:.6f}"
                )

            # Check if target reached
            if best_fitness >= target_fitness:
                print(f"Target fitness reached at generation {generation + 1}")
                return True

            # Preserve elite
            elite_indices = sorted(
                range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True
            )[: self.elite_size]
            elite = [self.population[i].copy() for i in elite_indices]

            # Selection and reproduction
            selected = self.selection(fitness_list)
            new_population = elite

            while len(new_population) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population[: self.population_size]

        print(
            f"Maximum generations ({max_generations}) reached. Best fitness: {self.best_fitness:.6f}"
        )
        return False
