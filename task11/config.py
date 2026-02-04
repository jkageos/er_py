"""Centralized configuration for Task 11."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PhysicsConfig:
    """Physics simulation parameters."""

    mass: float = 1.0
    friction: float = 0.5
    elasticity: float = 0.1
    damping: float = 0.95
    iterations: int = 10


@dataclass
class LidarConfig:
    """LiDAR sensor configuration."""

    num_rays: int = 5
    max_range: float = 200.0


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    width: int = 800
    height: int = 600
    robot_radius: float = 10.0
    max_sensor_range: float = 200.0
    maze_cols: int = 8
    maze_rows: int = 8
    wall_thickness: float = 8.0
    seed: Optional[int] = 42
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)

    # Adaptive parameters (computed if None)
    _max_speed: Optional[float] = None
    _eval_steps: Optional[int] = None

    @property
    def max_speed(self) -> float:
        if self._max_speed is None:
            self._max_speed, _ = compute_adaptive_params(self)
        return self._max_speed

    @property
    def eval_steps(self) -> int:
        if self._eval_steps is None:
            _, self._eval_steps = compute_adaptive_params(self)
        return self._eval_steps

    @property
    def n_lidar_rays(self) -> int:
        return self.lidar.num_rays


@dataclass
class EvolutionConfig:
    """Evolution algorithm configuration."""

    population_size: int = 100
    n_generations: int = 300
    n_hidden: int = 8
    n_hidden_layers: int = 2
    mutation_rate: float = 0.4
    mutation_sigma: float = 0.8
    tournament_size: int = 3
    elitism: int = 3
    iso_sigma: float = 0.02
    line_sigma: float = 0.2
    # CMA-ES options
    use_cmaes: bool = True
    num_emitters: int = 5
    emitter_batch_size: int = 30
    restart_threshold: float = 0.01


@dataclass
class NoveltyConfig:
    """Novelty search configuration."""

    archive_threshold: float = 0.005
    k_nearest: int = 15
    target_add_rate: float = 0.2
    adapt_rate: float = 0.02
    min_threshold: float = 0.001
    max_threshold: float = 0.03
    archive_max_size: int = 10000
    use_extended_behavior: bool = False


@dataclass
class OutputConfig:
    """Output and rendering configuration."""

    plots_dir: str = "plots"
    results_dir: str = "results"
    print_every: int = 10
    render_enabled: bool = False
    render_every: int = 1
    dpi: int = 150


@dataclass
class ParallelConfig:
    """Parallel processing configuration."""

    enabled: bool = True
    n_workers: Optional[int] = None


@dataclass
class Task11Config:
    """Complete Task 11 configuration."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    @property
    def maze_file(self) -> Path:
        return Path(self.output.results_dir) / "maze.pkl"

    @property
    def genome_file(self) -> Path:
        return Path(self.output.results_dir) / "task11_best_genome.pkl"


def compute_adaptive_params(
    env: EnvironmentConfig,
    dt: float = 1 / 60.0,
    path_complexity_factor: float = 2.5,
    safety_margin: float = 1.3,
) -> tuple[float, int]:
    """Compute adaptive max_speed and eval_steps based on maze geometry."""
    import math

    diagonal = math.sqrt(env.width**2 + env.height**2)
    complexity = 1 + (env.maze_cols * env.maze_rows) / 50
    estimated_path_length = diagonal * path_complexity_factor * min(complexity, 3.0)

    base_time = 10.0
    size_factor = (env.maze_cols * env.maze_rows) / 32
    target_time = base_time * max(1.0, size_factor) * safety_margin

    max_speed = estimated_path_length / target_time
    max_speed = max(50.0, min(500.0, max_speed))

    eval_steps = int(target_time / dt)
    eval_steps = max(300, min(3000, eval_steps))

    return max_speed, eval_steps


def load_config(config_path: str | Path | None = None) -> Task11Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return Task11Config()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    task11_raw = raw.get("task11", {})
    parallel_raw = raw.get("parallel", {})
    global_raw = raw.get("global", {})

    # Build nested configs
    env_raw = task11_raw.get("environment", {})
    physics = PhysicsConfig(**env_raw.get("physics", {}))
    lidar = LidarConfig(
        num_rays=env_raw.get("lidar", {}).get("num_rays", 5),
        max_range=env_raw.get("max_sensor_range", 200.0),
    )

    # Handle "auto" values
    max_speed = env_raw.get("max_speed")
    eval_steps = env_raw.get("eval_steps")

    environment = EnvironmentConfig(
        width=env_raw.get("width", 800),
        height=env_raw.get("height", 600),
        robot_radius=env_raw.get("robot_radius", 10.0),
        max_sensor_range=env_raw.get("max_sensor_range", 200.0),
        maze_cols=env_raw.get("maze_cols", 8),
        maze_rows=env_raw.get("maze_rows", 8),
        wall_thickness=env_raw.get("wall_thickness", 8.0),
        seed=env_raw.get("seed"),
        physics=physics,
        lidar=lidar,
        _max_speed=None if max_speed == "auto" else max_speed,
        _eval_steps=None if eval_steps == "auto" else eval_steps,
    )

    # Build evolution config, filtering to known fields
    evo_raw = task11_raw.get("evolution", {})
    evolution = EvolutionConfig(
        population_size=evo_raw.get("population_size", 100),
        n_generations=evo_raw.get("n_generations", 300),
        n_hidden=evo_raw.get("n_hidden", 8),
        n_hidden_layers=evo_raw.get("n_hidden_layers", 2),
        mutation_rate=evo_raw.get("mutation_rate", 0.4),
        mutation_sigma=evo_raw.get("mutation_sigma", 0.8),
        tournament_size=evo_raw.get("tournament_size", 3),
        elitism=evo_raw.get("elitism", 3),
        iso_sigma=evo_raw.get("iso_sigma", 0.02),
        line_sigma=evo_raw.get("line_sigma", 0.2),
        use_cmaes=evo_raw.get("use_cmaes", True),
        num_emitters=evo_raw.get("num_emitters", 5),
        emitter_batch_size=evo_raw.get("emitter_batch_size", 30),
        restart_threshold=evo_raw.get("restart_threshold", 0.01),
    )

    novelty = NoveltyConfig(**task11_raw.get("novelty", {}))

    output_raw = task11_raw.get("output", {})
    render_raw = task11_raw.get("rendering", {})
    output = OutputConfig(
        plots_dir=output_raw.get("plots_dir", "plots"),
        results_dir=output_raw.get("results_dir", "results"),
        print_every=output_raw.get("print_every", 10),
        render_enabled=render_raw.get("enabled", False),
        render_every=render_raw.get("render_every", 1),
        dpi=global_raw.get("dpi", 150),
    )

    parallel = ParallelConfig(
        enabled=parallel_raw.get("enabled", True),
        n_workers=parallel_raw.get("n_workers"),
    )

    return Task11Config(
        environment=environment,
        evolution=evolution,
        novelty=novelty,
        output=output,
        parallel=parallel,
    )


# Singleton for default config
_default_config: Task11Config | None = None


def get_config() -> Task11Config:
    """Get the default configuration (lazy loaded)."""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config


def reset_config() -> None:
    """Reset the cached config (useful after changing config file)."""
    global _default_config
    _default_config = None
