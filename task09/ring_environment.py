"""Ring environment for the behavioral attractor problem."""

import random
from typing import List, Tuple


class RingEnvironment:
    """Circular ring environment with numbered cells."""

    def __init__(self, cells_per_half: int = 20):
        """
        Initialize ring environment.

        Args:
            cells_per_half: Number of cells in each half of the ring
        """
        self.cells_per_half = cells_per_half
        self.total_cells = 2 * cells_per_half

        # Generate random numbering for each half
        left_numbers = list(range(cells_per_half))
        right_numbers = list(range(cells_per_half))
        random.shuffle(left_numbers)
        random.shuffle(right_numbers)

        # Combine: left half (cells 0-19) and right half (cells 20-39)
        self.cell_numbers = left_numbers + right_numbers

    def get_cell_number(self, position: int) -> int:
        """Get the number displayed at a given cell position."""
        return self.cell_numbers[position % self.total_cells]

    def is_left_half(self, position: int) -> bool:
        """Check if position is in the left half of the ring."""
        return (position % self.total_cells) < self.cells_per_half

    def move_clockwise(self, position: int) -> int:
        """Move one step clockwise."""
        return (position + 1) % self.total_cells

    def move_counterclockwise(self, position: int) -> int:
        """Move one step counterclockwise."""
        return (position - 1) % self.total_cells


class RingAgent:
    """Agent that navigates the ring environment."""

    def __init__(self, weights: List[float], threshold: float = 0.0):
        """
        Initialize agent with neural network weights.

        Args:
            weights: List of 20 weights, one for each input unit
            threshold: Threshold for action selection (>= threshold = CW, < threshold = CCW)
        """
        if len(weights) != 20:
            raise ValueError("Agent requires exactly 20 weights")
        self.weights = weights
        self.threshold = threshold

    @staticmethod
    def activation(x: float) -> float:
        """
        Activation function: tanh for bounded output.

        Args:
            x: Input value

        Returns:
            Activated value in range [-1, 1]
        """
        import math

        return math.tanh(x)

    def decide_action(self, cell_number: int) -> str:
        """
        Decide whether to move clockwise or counterclockwise.

        Args:
            cell_number: Number of the current cell (0-19)

        Returns:
            "CW" for clockwise, "CCW" for counterclockwise
        """
        # Only one input is active (value 1.0), others are 0
        weighted_input = self.weights[cell_number]
        output = self.activation(weighted_input)

        return "CW" if output >= self.threshold else "CCW"

    def simulate(
        self, environment: RingEnvironment, steps: int, start_position: int
    ) -> Tuple[int, List[int]]:
        """
        Simulate agent behavior for a number of steps.

        Args:
            environment: Ring environment
            steps: Number of steps to simulate
            start_position: Initial position on the ring

        Returns:
            Tuple of (time_in_left_half, trajectory)
        """
        position = start_position
        time_in_left = 0
        trajectory = [position]

        for _ in range(steps):
            # Check if in left half
            if environment.is_left_half(position):
                time_in_left += 1

            # Get current cell number and decide action
            cell_number = environment.get_cell_number(position)
            action = self.decide_action(cell_number)

            # Move
            if action == "CW":
                position = environment.move_clockwise(position)
            else:
                position = environment.move_counterclockwise(position)

            trajectory.append(position)

        return time_in_left, trajectory
