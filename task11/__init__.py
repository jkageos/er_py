"""Task 11: Maze Exploration with Novelty Search."""

from task11.ann import RecurrentANN, get_genome_size
from task11.config import (
    EnvironmentConfig,
    EvolutionConfig,
    NoveltyConfig,
    OutputConfig,
    ParallelConfig,
    Task11Config,
    get_config,
    load_config,
)
from task11.controller import MazeController
from task11.environment import MazeEnvironment
from task11.evaluation import PopulationEvaluator, evaluate_genome
from task11.evolution import NoveltySearchEvolution
from task11.ga import GeneticAlgorithm
from task11.maze_generator import MazeGenerator, generate_maze
from task11.metrics import (
    EpisodeMetrics,
    EvolutionHistory,
    GenerationMetrics,
    compute_fitness_proxy,
    compute_generation_metrics,
)
from task11.novelty_archive import NoveltyArchive
from task11.plotting import Task11Plotter
from task11.recorder import TrajectoryRecorder, VideoRecorder, record_controller_run
from task11.robot_agent import RobotAgent
from task11.sensors import BumperSensors, LiDAR

__all__ = [
    # Config
    "Task11Config",
    "EnvironmentConfig",
    "EvolutionConfig",
    "NoveltyConfig",
    "OutputConfig",
    "ParallelConfig",
    "get_config",
    "load_config",
    # Core
    "MazeEnvironment",
    "MazeController",
    "RecurrentANN",
    "get_genome_size",
    # Evolution
    "NoveltySearchEvolution",
    "GeneticAlgorithm",
    "NoveltyArchive",
    "PopulationEvaluator",
    "evaluate_genome",
    # Metrics
    "EpisodeMetrics",
    "GenerationMetrics",
    "EvolutionHistory",
    "compute_generation_metrics",
    "compute_fitness_proxy",
    # Plotting
    "Task11Plotter",
    # Recording
    "TrajectoryRecorder",
    "VideoRecorder",
    "record_controller_run",
    # Environment
    "MazeGenerator",
    "generate_maze",
    "RobotAgent",
    "LiDAR",
    "BumperSensors",
]
