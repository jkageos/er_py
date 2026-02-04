"""Robot controller using RecurrentANN."""

import numpy as np

from task11.ann import RecurrentANN


class MazeController:
    """Wraps RecurrentANN for maze navigation."""

    # Default input size (can be overridden)
    # With 5 LiDAR rays + 2 bumpers = 7 inputs
    DEFAULT_N_LIDAR = 5
    N_BUMPERS = 2
    N_OUTPUTS = 2  # left motor, right motor

    def __init__(
        self,
        n_hidden: int = 8,
        n_hidden_layers: int = 2,
        genome: np.ndarray | None = None,
        n_lidar_rays: int = 5,
    ):
        self.n_lidar_rays = n_lidar_rays
        self.n_inputs = n_lidar_rays + self.N_BUMPERS
        self.n_outputs = self.N_OUTPUTS
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers

        self.ann = RecurrentANN(
            self.n_inputs,
            n_hidden,
            self.n_outputs,
            genome=genome,
            n_hidden_layers=n_hidden_layers,
        )

    def reset(self) -> None:
        """Reset controller state."""
        self.ann.reset()

    def forward(self, sensors: list[float]) -> tuple[float, float]:
        """Forward pass through controller.

        Args:
            sensors: LiDAR readings + bumper readings
                     (n_lidar + 2 bumpers)

        Returns:
            (left_motor, right_motor) in [-1, 1]
        """
        outputs = self.ann.forward(sensors)
        left_motor = float(np.clip(outputs[0], -1, 1))
        right_motor = float(np.clip(outputs[1], -1, 1))
        return left_motor, right_motor

    @classmethod
    def get_genome_size(
        cls, n_hidden: int = 8, n_hidden_layers: int = 2, n_lidar_rays: int = 5
    ) -> int:
        """Get the genome size for given architecture."""
        from task11.ann import get_genome_size

        n_inputs = n_lidar_rays + cls.N_BUMPERS
        return get_genome_size(n_inputs, n_hidden, cls.N_OUTPUTS, n_hidden_layers)
