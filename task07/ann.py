from typing import Optional

import numpy as np


class ANN:
    """Artificial Neural Network with 2 input, 2 hidden, and 1 output neuron."""

    def __init__(
        self, weights: Optional[np.ndarray] = None, biases: Optional[np.ndarray] = None
    ):
        """
        Initialize ANN with weights and biases.

        Args:
            weights: Array of shape (6,) containing all weights [w_ih0, w_ih1, w_ih2, w_ih3, w_ho0, w_ho1]
            biases: Array of shape (3,) containing biases [b_h0, b_h1, b_o]
        """
        if weights is None:
            self.weights = np.random.uniform(-1, 1, 6)
        else:
            self.weights = np.array(weights, dtype=np.float64)

        if biases is None:
            self.biases = np.random.uniform(-1, 1, 3)
        else:
            self.biases = np.array(biases, dtype=np.float64)

    @staticmethod
    def sigmoid(x: float) -> float:
        """
        Sigmoid activation function: φ(x) = 2/(1+exp(-2x)) - 1

        Args:
            x: Input value

        Returns:
            Activated value in range [-1, 1]
        """
        return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

    def forward(self, a: float, b: float) -> float:
        """
        Forward pass through the network.

        Args:
            a: First input
            b: Second input

        Returns:
            Output of the network
        """
        # Input to hidden layer
        h0_input = self.weights[0] * a + self.weights[1] * b + self.biases[0]
        h1_input = self.weights[2] * a + self.weights[3] * b + self.biases[1]

        # Hidden layer activation
        h0 = self.sigmoid(h0_input)
        h1 = self.sigmoid(h1_input)

        # Hidden to output layer
        o_input = self.weights[4] * h0 + self.weights[5] * h1 + self.biases[2]

        # Output activation
        output = self.sigmoid(o_input)

        return output

    def evaluate_grid(self, resolution: int = 50) -> np.ndarray:
        """
        Evaluate the ANN over a grid of inputs [0, 1]².

        Args:
            resolution: Grid resolution

        Returns:
            2D array of outputs
        """
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.forward(X[i, j], Y[i, j])

        return Z

    def get_genes(self) -> np.ndarray:
        """Get concatenated weights and biases as gene representation."""
        return np.concatenate([self.weights, self.biases])

    def set_genes(self, genes: np.ndarray) -> None:
        """Set weights and biases from gene representation."""
        self.weights = genes[:6].copy()
        self.biases = genes[6:9].copy()
