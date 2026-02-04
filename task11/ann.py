"""Recurrent Artificial Neural Network for maze navigation."""

import numpy as np


class RecurrentANN:
    """Simple recurrent neural network with tanh activation.

    Architecture matches Kheperax MLP: input -> hidden1 -> hidden2 -> output
    All layers use tanh activation for outputs in [-1, 1].
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        genome: np.ndarray | None = None,
        n_hidden_layers: int = 2,
    ):
        """Initialize the recurrent ANN.

        Args:
            n_inputs: Number of input neurons
            n_hidden: Number of neurons per hidden layer
            n_outputs: Number of output neurons
            genome: Flat array of weights and biases
            n_hidden_layers: Number of hidden layers
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers

        # Initialize weights and biases
        self._init_weights(genome)

        # Hidden state for recurrence (optional, can be disabled)
        self.hidden_state = [np.zeros(n_hidden) for _ in range(n_hidden_layers)]
        self.use_recurrence = False  # Disable recurrence like Kheperax

    def _init_weights(self, genome: np.ndarray | None = None) -> None:
        """Initialize weights from genome or randomly."""
        self.weights = []
        self.biases = []

        layer_sizes = (
            [self.n_inputs] + [self.n_hidden] * self.n_hidden_layers + [self.n_outputs]
        )

        if genome is not None:
            # Unpack genome into weights and biases
            idx = 0
            for i in range(len(layer_sizes) - 1):
                in_size = layer_sizes[i]
                out_size = layer_sizes[i + 1]

                # Weights
                w_size = in_size * out_size
                w = genome[idx : idx + w_size].reshape(in_size, out_size)
                self.weights.append(w)
                idx += w_size

                # Biases
                b = genome[idx : idx + out_size]
                self.biases.append(b)
                idx += out_size
        else:
            # Random initialization with Xavier/Glorot
            for i in range(len(layer_sizes) - 1):
                in_size = layer_sizes[i]
                out_size = layer_sizes[i + 1]

                # Xavier initialization
                limit = np.sqrt(6.0 / (in_size + out_size))
                w = np.random.uniform(-limit, limit, (in_size, out_size))
                b = np.zeros(out_size)

                self.weights.append(w)
                self.biases.append(b)

    def reset(self) -> None:
        """Reset hidden state."""
        self.hidden_state = [
            np.zeros(self.n_hidden) for _ in range(self.n_hidden_layers)
        ]

    def forward(self, inputs: list[float]) -> np.ndarray:
        """Forward pass through the network.

        Args:
            inputs: Input values (sensor readings)

        Returns:
            Output values (motor commands) in [-1, 1]
        """
        x = np.array(inputs, dtype=np.float64)

        # Pass through each layer
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            # tanh activation for all layers (including output)
            x = np.tanh(x)

        return x

    def get_genome(self) -> np.ndarray:
        """Get flattened genome."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.flatten())
            parts.append(b)
        return np.concatenate(parts)

    def set_genome(self, genome: np.ndarray) -> None:
        """Set weights from genome."""
        self._init_weights(genome)


def get_genome_size(
    n_inputs: int,
    n_hidden: int,
    n_outputs: int,
    n_hidden_layers: int = 2,
) -> int:
    """Calculate total genome size for given architecture."""
    layer_sizes = [n_inputs] + [n_hidden] * n_hidden_layers + [n_outputs]

    total = 0
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        total += in_size * out_size  # weights
        total += out_size  # biases

    return total
