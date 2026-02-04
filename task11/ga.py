"""Genetic algorithm for novelty search evolution."""

from typing import List

import numpy as np


class CMAESEmitter:
    """CMA-ES based emitter for novelty search (inspired by pyribs)."""

    def __init__(
        self,
        genome_size: int,
        x0: np.ndarray | None = None,
        sigma0: float = 0.5,
        batch_size: int = 30,
    ):
        self.genome_size = genome_size
        self.sigma0 = sigma0
        self.batch_size = batch_size

        # Initialize mean
        self.mean = x0 if x0 is not None else np.zeros(genome_size)
        self.sigma = sigma0

        # CMA-ES parameters
        self.C = np.eye(genome_size)  # Covariance matrix
        self.pc = np.zeros(genome_size)  # Evolution path for C
        self.ps = np.zeros(genome_size)  # Evolution path for sigma

        # Strategy parameters
        self.mu = max(1, batch_size // 2)  # Number of parents
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)

        # Adaptation parameters
        self.cc = 4.0 / (genome_size + 4)
        self.cs = (self.mueff + 2) / (genome_size + self.mueff + 3)
        self.c1 = 2 / ((genome_size + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2
            * (self.mueff - 2 + 1 / self.mueff)
            / ((genome_size + 2) ** 2 + self.mueff),
        )
        self.damps = (
            1 + 2 * max(0, np.sqrt((self.mueff - 1) / (genome_size + 1)) - 1) + self.cs
        )
        self.chiN = np.sqrt(genome_size) * (
            1 - 1 / (4 * genome_size) + 1 / (21 * genome_size**2)
        )

    def ask(self) -> List[np.ndarray]:
        """Generate batch of solutions."""
        # Eigendecomposition for sampling
        D, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D, 1e-20))

        solutions = []
        for _ in range(self.batch_size):
            z = np.random.randn(self.genome_size)
            y = B @ (D * z)
            x = self.mean + self.sigma * y
            solutions.append(x)

        self._last_solutions = solutions
        return solutions

    def tell(self, solutions: List[np.ndarray], novelty_scores: List[float]) -> None:
        """Update distribution based on novelty scores."""
        if len(solutions) < self.mu:
            # Not enough solutions to update
            return

        # Rank solutions by novelty (higher is better)
        indices = np.argsort(novelty_scores)[::-1]

        # Select top mu solutions
        selected = [solutions[i] for i in indices[: self.mu]]

        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum(
            [self.weights[i] * selected[i] for i in range(self.mu)], axis=0
        )

        # Update evolution paths
        D, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D, 1e-20))
        invsqrtC = B @ np.diag(1.0 / D) @ B.T

        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * invsqrtC @ (self.mean - old_mean) / self.sigma

        hsig = (
            np.linalg.norm(self.ps)
            / np.sqrt(1 - (1 - self.cs) ** (2 * (self.batch_size + 1)))
            < (1.4 + 2 / (self.genome_size + 1)) * self.chiN
        )

        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * (self.mean - old_mean) / self.sigma

        # Update covariance matrix
        artmp = (np.array(selected) - old_mean) / self.sigma
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1
            * (
                np.outer(self.pc, self.pc)
                + (1 - hsig) * self.cc * (2 - self.cc) * self.C
            )
            + self.cmu * (artmp.T @ np.diag(self.weights) @ artmp)
        )

        # Update step size
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )
        self.sigma = np.clip(self.sigma, 1e-20, 1e20)


class GeneticAlgorithm:
    """Genetic algorithm with multiple CMA-ES emitters for novelty search."""

    def __init__(
        self,
        genome_size: int,
        population_size: int = 150,
        mutation_rate: float = 0.3,
        mutation_sigma: float = 0.5,
        tournament_size: int = 3,
        elitism: int = 5,
        iso_sigma: float = 0.01,
        line_sigma: float = 0.1,
        num_emitters: int = 5,
        use_cmaes: bool = True,
        emitter_batch_size: int = 30,
    ):
        """Initialize genetic algorithm."""
        self.genome_size = genome_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma
        self.num_emitters = num_emitters
        self.emitter_batch_size = emitter_batch_size
        self.emitters: List[CMAESEmitter] = []
        self._use_cmaes = use_cmaes
        self._generation = 0

    def initialize(self) -> List[np.ndarray]:
        """Initialize population with diverse random genomes."""
        population = []

        # Create diverse initial population with different initializations
        for i in range(self.population_size):
            # Vary the initialization scale to get different behaviors
            scale = 0.5 + (i / self.population_size) * 1.5  # Range: 0.5 to 2.0
            genome = np.random.randn(self.genome_size) * scale
            population.append(genome)

        # Initialize CMA-ES emitters with diverse starting points
        if self._use_cmaes:
            batch_per_emitter = max(2, self.population_size // self.num_emitters)
            self.emitters = []
            for i in range(self.num_emitters):
                # Each emitter starts from a different random point
                # with different initial sigma for diversity
                x0 = np.random.randn(self.genome_size) * (0.5 + i * 0.3)
                sigma0 = 0.3 + i * 0.2  # Range: 0.3 to 1.1
                self.emitters.append(
                    CMAESEmitter(
                        self.genome_size,
                        x0=x0,
                        sigma0=sigma0,
                        batch_size=batch_per_emitter,
                    )
                )

        return population

    def create_next_generation(
        self,
        population: List[np.ndarray],
        fitness: List[float],
    ) -> List[np.ndarray]:
        """Create next generation using CMA-ES emitters."""
        self._generation += 1

        if self._use_cmaes and self.emitters:
            return self._cmaes_generation(population, fitness)
        return self._simple_generation(population, fitness)

    def _cmaes_generation(
        self,
        population: List[np.ndarray],
        fitness: List[float],
    ) -> List[np.ndarray]:
        """Generate using CMA-ES emitters."""
        new_population: List[np.ndarray] = []

        # Tell emitters about previous results
        batch_size = max(1, len(population) // self.num_emitters)
        for i, emitter in enumerate(self.emitters):
            start = i * batch_size
            end = min(start + batch_size, len(population))
            if start < len(population):
                emitter.tell(population[start:end], fitness[start:end])

        # Ask emitters for new solutions
        for emitter in self.emitters:
            new_population.extend(emitter.ask())

        # Add some random individuals for diversity (especially early on)
        num_random = max(5, self.population_size // 10)
        for _ in range(num_random):
            scale = np.random.uniform(0.5, 2.0)
            new_population.append(np.random.randn(self.genome_size) * scale)

        # Ensure we have enough individuals
        while len(new_population) < self.population_size:
            new_population.append(np.random.randn(self.genome_size) * 1.0)

        return new_population[: self.population_size]

    def _simple_generation(
        self,
        population: List[np.ndarray],
        fitness: List[float],
    ) -> List[np.ndarray]:
        """Fallback to simple GA generation."""
        new_population: List[np.ndarray] = []

        # Elitism: keep best (most novel) individuals
        if self.elitism > 0:
            elite_indices = np.argsort(fitness)[-self.elitism :]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

        # Also inject some random individuals for diversity
        num_random = max(1, self.population_size // 20)  # 5% random
        for _ in range(num_random):
            random_genome = np.random.randn(self.genome_size) * 1.0
            new_population.append(random_genome)

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(population, fitness)
            parent2 = self._tournament_select(population, fitness)

            # Use ISO-Line variation
            child = self._iso_line_variation(parent1, parent2)

            # Always mutate for exploration
            child = self._mutate(child)

            new_population.append(child)

        return new_population[: self.population_size]

    def _tournament_select(
        self,
        population: List[np.ndarray],
        fitness: List[float],
    ) -> np.ndarray:
        """Select an individual using tournament selection."""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        best_idx = indices[np.argmax([fitness[i] for i in indices])]
        return population[best_idx].copy()

    def _iso_line_variation(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> np.ndarray:
        """ISO-Line variation operator."""
        direction = parent2 - parent1
        iso_noise = np.random.randn(self.genome_size) * self.iso_sigma
        line_noise = np.random.randn() * self.line_sigma * direction
        child = parent1 + iso_noise + line_noise
        return child

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """Mutate genome with Gaussian noise."""
        noise = np.random.randn(self.genome_size) * self.mutation_sigma
        mask = np.random.random(self.genome_size) < self.mutation_rate
        genome = genome + noise * mask
        return genome

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> np.ndarray:
        """Uniform crossover between two parents."""
        mask = np.random.random(self.genome_size) < 0.5
        child = np.where(mask, parent1, parent2)
        return child
