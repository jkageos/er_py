"""Main novelty search evolution orchestrator."""

import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from task11.config import Task11Config, get_config
from task11.controller import MazeController
from task11.environment import MazeEnvironment
from task11.evaluation import BehaviorType, PopulationEvaluator
from task11.ga import GeneticAlgorithm
from task11.metrics import (
    EvolutionHistory,
    compute_generation_metrics,
)
from task11.novelty_archive import NoveltyArchive
from task11.plotting import Task11Plotter
from utils.graceful_exit import GracefulExitHandler
from utils.parallel import get_optimal_workers


class NoveltySearchEvolution:
    """Evolve maze exploration using Novelty Search."""

    def __init__(self, config: Task11Config | None = None):
        """Initialize novelty search evolution."""
        self.config = config or get_config()
        self._setup_directories()
        self._init_components()

        # State
        self._exit_handler = GracefulExitHandler(
            message="\n\nCtrl+C detected! Finishing current generation...\n"
        )
        self._interrupted = False

        # History tracking
        self.history = EvolutionHistory()

        # Store trajectories for exploration map (every generation)
        self._all_trajectories: List[List[Tuple[float, float]]] = []

        # Store best trajectory for final plotting
        self._best_trajectory: List[Tuple[float, float]] = []
        self._best_goal_trajectory: List[Tuple[float, float]] | None = None

    def _setup_directories(self) -> None:
        """Create output directories."""
        Path(self.config.output.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output.plots_dir).mkdir(parents=True, exist_ok=True)

    def _init_components(self) -> None:
        """Initialize evolution components."""
        env = self.config.environment
        evo = self.config.evolution
        nov = self.config.novelty

        # Get number of lidar rays from config
        n_lidar_rays = env.n_lidar_rays

        self.genome_size = MazeController.get_genome_size(
            evo.n_hidden, evo.n_hidden_layers, n_lidar_rays
        )

        self.ga = GeneticAlgorithm(
            genome_size=self.genome_size,
            population_size=evo.population_size,
            mutation_rate=evo.mutation_rate,
            mutation_sigma=evo.mutation_sigma,
            tournament_size=evo.tournament_size,
            elitism=evo.elitism,
            iso_sigma=evo.iso_sigma,
            line_sigma=evo.line_sigma,
            num_emitters=evo.num_emitters,
            use_cmaes=evo.use_cmaes,
            emitter_batch_size=evo.emitter_batch_size,
        )

        self.archive = NoveltyArchive(
            env_width=env.width,
            env_height=env.height,
            k_nearest=nov.k_nearest,
            archive_threshold=nov.archive_threshold,
            target_add_rate=nov.target_add_rate,
            adapt_rate=nov.adapt_rate,
            min_threshold=nov.min_threshold,
            max_threshold=nov.max_threshold,
            max_size=nov.archive_max_size,
        )

        self.evaluator = PopulationEvaluator(self.config)
        self.plotter = Task11Plotter(self.config.output.plots_dir)

        # Store goal position for metrics
        margin = self.config.environment.wall_thickness
        cell_width = (env.width - 2 * margin) / env.maze_cols
        cell_height = (env.height - 2 * margin) / env.maze_rows
        self._goal_pos = (
            env.width - margin - cell_width / 2,
            env.height - margin - cell_height / 2,
        )
        self._start_pos = (
            margin + cell_width / 2,
            margin + cell_height / 2,
        )

    def _create_maze(self) -> None:
        """Create and save initial maze, set up archive mask."""
        env = MazeEnvironment(
            config=self.config.environment,
            maze_file=self.config.maze_file,
            render=False,
            spawn_robot=False,
        )

        self.archive.set_reachable_mask(
            maze_grid=env.maze_grid,
            wall_thickness=self.config.environment.wall_thickness,
            robot_radius=self.config.environment.robot_radius,
        )

        # Store maze grid for plotting
        self._maze_grid = env.maze_grid

        env.close()

    def _print_config(self) -> None:
        """Print evolution configuration."""
        env = self.config.environment
        evo = self.config.evolution
        par = self.config.parallel

        print("Starting Novelty Search Evolution")
        print("=" * 60)
        print(f"Population size: {evo.population_size}")
        print(f"Generations: {evo.n_generations}")
        print(f"Genome size: {self.genome_size}")
        print(f"Environment: {env.width}x{env.height} pixels")
        print(f"Maze: {env.maze_cols}x{env.maze_rows} cells")
        print("-" * 60)
        print("Adaptive Parameters:")
        print(f"  Max speed: {env.max_speed:.1f} pixels/second")
        print(f"  Eval steps: {env.eval_steps} steps")
        print(f"  Episode time: {env.eval_steps / 60:.1f} seconds")
        print("-" * 60)
        print(f"Archive threshold: {self.archive.archive_threshold:.1f} pixels")
        print(f"K-nearest neighbors: {self.archive.k_nearest}")
        print(f"Reachable cells: {self.archive.get_num_reachable_cells()}")
        print(f"Parallel enabled: {par.enabled}")
        if par.enabled:
            print(f"  Workers: {get_optimal_workers(par.n_workers)}")
        print("-" * 60)
        print("Press Ctrl+C to gracefully stop")
        print("=" * 60)

    def _generation_step(
        self,
        population: List[np.ndarray],
        generation: int,
    ) -> Tuple[List[np.ndarray], np.ndarray, float]:
        """Execute one generation."""
        # Evaluate
        behaviors: List[BehaviorType]
        behaviors, trajectories, goal_reached = self.evaluator.evaluate(population)

        # Store all trajectories for exploration map (every generation)
        self._all_trajectories.extend(trajectories)

        # Compute novelty
        novelty_scores = self.archive.compute_novelty_batch(behaviors)

        # Update archive with genomes
        num_added = self.archive.update_with_genomes(
            behaviors, novelty_scores, population
        )
        self.archive.adapt_threshold(num_added, len(population))

        # Compute generation metrics
        gen_metrics = compute_generation_metrics(
            generation=generation,
            behaviors=behaviors,
            trajectories=trajectories,
            goal_reached=goal_reached,
            novelty_scores=novelty_scores,
            archive_size=len(self.archive),
            coverage=self.archive.compute_coverage(),
            cells_visited=self.archive.get_num_cells_visited(),
            goal_pos=self._goal_pos,
        )
        self.history.add_generation(gen_metrics)

        # Track first goal-reaching genome and its trajectory
        num_goal_reached = sum(goal_reached)
        if num_goal_reached > 0 and self.history.best_goal_genome is None:
            for i, reached in enumerate(goal_reached):
                if reached:
                    self.history.best_goal_genome = population[i].copy()
                    self.history.best_goal_generation = generation + 1
                    self._best_goal_trajectory = trajectories[i].copy()
                    print(
                        f"\n** GOAL REACHED for first time at generation {generation + 1}!"
                    )
                    break

        # Track best novelty genome and trajectory
        novelty_arr = np.array(novelty_scores)
        best_idx = int(np.argmax(novelty_arr))
        best_novelty = novelty_scores[best_idx]
        best_genome = population[best_idx].copy()

        if best_novelty > self.history.best_novelty_score:
            self.history.best_novelty_score = best_novelty
            self.history.best_novelty_genome = best_genome.copy()
            self._best_trajectory = trajectories[best_idx].copy()

        # Print progress
        if (generation + 1) % self.config.output.print_every == 0:
            goal_str = (
                f"Goals: {num_goal_reached:3d}"
                if num_goal_reached > 0
                else "Goals:   0"
            )

            print(
                f"Gen {generation + 1:4d} | "
                f"Avg: {gen_metrics.avg_novelty:7.1f} | "
                f"Max: {gen_metrics.max_novelty:7.1f} | "
                f"Archive: {gen_metrics.archive_size:5d} | "
                f"Coverage: {gen_metrics.coverage:5.1%} | "
                f"Dist: {gen_metrics.best_distance_to_goal:6.0f} | "
                f"{goal_str}"
            )

        # Create next generation
        new_population = self.ga.create_next_generation(population, novelty_scores)

        # Check for stagnation and trigger restart
        if len(self.history.generations) > 50:
            recent_coverage = self.history.coverage_history[-50:]
            improvement = recent_coverage[-1] - recent_coverage[0]

            if improvement < 0.01:
                print("  Stagnation detected - restarting emitters from diverse points")
                self._restart_from_diverse_points()

        return new_population, best_genome, best_novelty

    def _restart_from_diverse_points(self) -> None:
        """Restart emitters from diverse archived solutions."""
        diverse_genomes = self.archive.get_diverse_genomes(self.ga.num_emitters)

        if diverse_genomes and self.ga.emitters:
            for emitter, genome in zip(self.ga.emitters, diverse_genomes):
                emitter.mean = genome.copy()
                emitter.sigma = emitter.sigma0

    def _save_results(self, best_genome: np.ndarray | None) -> None:
        """Save best genome, trajectory data, and generate all plots."""
        # Prefer goal-reaching genome if available
        genome_to_save = (
            self.history.best_goal_genome
            if self.history.best_goal_genome is not None
            else best_genome
        )

        # Choose best trajectory
        trajectory_to_save = (
            self._best_goal_trajectory
            if self._best_goal_trajectory is not None
            else self._best_trajectory
        )
        goal_reached = self._best_goal_trajectory is not None

        if genome_to_save is not None:
            with open(self.config.genome_file, "wb") as f:
                pickle.dump(genome_to_save, f)
            if self.history.best_goal_genome is not None:
                print(f"\nSaved goal-reaching genome to {self.config.genome_file}")
            else:
                print(f"\nSaved best genome to {self.config.genome_file}")

        # Save trajectory data for future plotting
        trajectory_file = Path(self.config.output.results_dir) / "best_trajectory.pkl"
        with open(trajectory_file, "wb") as f:
            pickle.dump(
                {
                    "trajectory": trajectory_to_save,
                    "goal_reached": goal_reached,
                    "genome": genome_to_save,
                },
                f,
            )
        print(f"Saved trajectory data to {trajectory_file}")

        # Save history
        history_file = Path(self.config.output.results_dir) / "evolution_history.pkl"
        with open(history_file, "wb") as f:
            pickle.dump(self.history, f)
        print(f"Saved evolution history to {history_file}")

        # Save all trajectories for exploration map
        if self._all_trajectories:
            trajectories_file = (
                Path(self.config.output.results_dir) / "all_trajectories.pkl"
            )
            with open(trajectories_file, "wb") as f:
                pickle.dump(self._all_trajectories, f)
            print(
                f"Saved {len(self._all_trajectories)} trajectories to {trajectories_file}"
            )

        # Generate plots
        if self.history.generations:
            self.plotter.plot_fitness_progress(
                self.history,
                save_name="task11_fitness_progress.png",
            )

            self.plotter.plot_novelty_metrics(
                self.history,
                save_name="task11_novelty_metrics.png",
            )

            # Plot best trajectory
            if trajectory_to_save:
                self.plotter.plot_best_trajectory(
                    trajectory=trajectory_to_save,
                    start_pos=self._start_pos,
                    goal_pos=self._goal_pos,
                    goal_reached=goal_reached,
                    width=self.config.environment.width,
                    height=self.config.environment.height,
                    maze_grid=self._maze_grid,
                    wall_thickness=self.config.environment.wall_thickness,
                    save_name="task11_best_trajectory.png",
                )

            # Plot exploration map (scatter plot of end positions)
            if self._all_trajectories:
                self.plotter.plot_exploration_map(
                    self._all_trajectories,
                    self.config.environment.width,
                    self.config.environment.height,
                    maze_grid=self._maze_grid,
                    wall_thickness=self.config.environment.wall_thickness,
                    save_name="task11_exploration_map.png",
                )

    def run(self) -> Tuple[np.ndarray | None, float]:
        """Run the evolution.

        Returns:
            Tuple of (best_genome, best_novelty)
        """
        self._print_config()
        self._create_maze()

        population = self.ga.initialize()
        best_genome: np.ndarray | None = None
        best_novelty = 0.0

        start_time = time.time()

        with self._exit_handler:
            for generation in range(self.config.evolution.n_generations):
                if self._exit_handler.should_exit:
                    self._interrupted = True
                    print(f"\nStopping at generation {generation}")
                    break

                population, gen_best_genome, gen_best_novelty = self._generation_step(
                    population, generation
                )

                if gen_best_novelty > best_novelty:
                    best_novelty = gen_best_novelty
                    best_genome = gen_best_genome.copy()

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("Evolution Complete!" + (" (interrupted)" if self._interrupted else ""))
        print(f"Generations: {len(self.history.generations)}")
        print(f"Best novelty: {best_novelty:.2f}")
        if self.history.coverage_history:
            print(f"Final coverage: {self.history.coverage_history[-1]:.1%}")
        print(f"Archive size: {len(self.archive)}")

        # Report goal statistics
        total_goals = sum(self.history.goals_reached_history)
        if total_goals > 0:
            print(f"Total goal reaches: {total_goals}")
            if self.history.best_goal_generation:
                print(f"First goal at generation: {self.history.best_goal_generation}")

        print(f"Time: {elapsed:.1f}s")
        print("=" * 60)

        self._save_results(best_genome)

        return (
            self.history.best_goal_genome
            if self.history.best_goal_genome is not None
            else best_genome,
            best_novelty,
        )

    @property
    def interrupted(self) -> bool:
        return self._interrupted
