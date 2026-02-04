"""Robot agent combining controller, sensors, and kinematics."""

import math
from typing import List, Tuple

import numpy as np
import pymunk

from task11.controller import MazeController
from task11.sensors import BumperSensors, LiDAR


class RobotAgent:
    """2-wheel differential drive robot with LiDAR and bumpers."""

    def __init__(
        self,
        x: float,
        y: float,
        robot_radius: float = 15.0,
        max_sensor_range: float = 100.0,
        max_speed: float = 100.0,
        controller: MazeController | None = None,
        physics: dict | None = None,
        lidar_config: dict | None = None,
        initial_angle: float | None = None,
    ):
        """Initialize robot agent."""
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.physics = physics or {}
        self.lidar_config = lidar_config or {}

        # Physics body
        mass = float(self.physics.get("mass", 1.0))
        moment = pymunk.moment_for_circle(mass, 0, robot_radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = (x, y)
        self.body.angle = initial_angle if initial_angle is not None else 0.0
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

        # Robot shape
        self.shape = pymunk.Circle(self.body, robot_radius)
        self.shape.friction = float(self.physics.get("friction", 0.7))
        self.shape.elasticity = float(self.physics.get("elasticity", 0.0))
        self.shape.collision_type = 1
        self.shape.filter = pymunk.ShapeFilter(group=1)

        # LiDAR sensor
        num_rays = int(self.lidar_config.get("num_rays", 3))
        self.lidar = LiDAR(
            max_range=max_sensor_range,
            num_rays=num_rays,
        )

        # Bumper sensors
        self.bumpers = BumperSensors()

        # Controller
        self.controller = controller or MazeController()

        # Trajectory history
        self._trajectory: np.ndarray = np.array([[x, y]], dtype=np.float64)

    def reset(self) -> None:
        """Reset agent state."""
        self.controller.reset()
        pos = self.body.position
        self._trajectory = np.array([[pos.x, pos.y]], dtype=np.float64)

    def update(self, space: pymunk.Space, dt: float = 1 / 60.0) -> None:
        """Update agent: sense, think, act."""
        # Update sensors
        self.lidar.update(self.body, space, self.shape, self.robot_radius)
        self.bumpers.update(self.body, self.shape, space)

        # Get sensor readings (already normalized to [0, 1])
        lidar_readings = self.lidar.get_readings()
        bumper_readings = self.bumpers.get_readings()

        # Combine inputs: 3 LiDAR + 2 bumpers = 5 inputs
        # LiDAR: 0 = obstacle close, 1 = clear
        # Bumpers: 0 = not touching, 1 = touching
        sensor_values = lidar_readings + bumper_readings

        # Get controller output
        left_motor, right_motor = self.controller.forward(sensor_values)

        # Clamp outputs to [-1, 1]
        left_motor = max(-1.0, min(1.0, left_motor))
        right_motor = max(-1.0, min(1.0, right_motor))

        # Differential drive kinematics
        # Average of motors gives forward speed
        # Difference gives rotation
        forward_speed = (left_motor + right_motor) / 2.0 * self.max_speed

        # Wheel base affects turning rate
        wheel_base = self.robot_radius * 2
        angular_speed = (right_motor - left_motor) / wheel_base * self.max_speed

        # Apply movement using velocity (kinematic control)
        direction = pymunk.Vec2d(math.cos(self.body.angle), math.sin(self.body.angle))
        self.body.velocity = direction * forward_speed
        self.body.angular_velocity = angular_speed

        # Apply damping manually for smoother physics
        damping = float(self.physics.get("damping", 0.95))
        self.body.velocity = self.body.velocity * damping

        # Step physics with substeps for stability
        substeps = 4
        sub_dt = dt / substeps
        for _ in range(substeps):
            space.step(sub_dt)

        # Record trajectory
        pos = self.body.position
        new_pos = np.array([[pos.x, pos.y]])
        self._trajectory = np.vstack([self._trajectory, new_pos])

    def get_position(self) -> Tuple[float, float]:
        """Get robot position."""
        pos = self.body.position
        return (pos.x, pos.y)

    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get trajectory history."""
        return [(float(x), float(y)) for x, y in self._trajectory]

    @staticmethod
    def compute_initial_angle(
        maze_grid: List[List[dict]],
        robot_row: int,
        robot_col: int,
    ) -> float:
        """Compute initial angle to face an open direction."""
        if not maze_grid or robot_row < 0 or robot_col < 0:
            return 0.0

        if robot_row >= len(maze_grid) or robot_col >= len(maze_grid[0]):
            return 0.0

        cell = maze_grid[robot_row][robot_col]

        directions = {
            "E": 0.0,
            "S": math.pi / 2,
            "W": math.pi,
            "N": -math.pi / 2,
        }

        priority = ["E", "S", "W", "N"]

        for direction in priority:
            if not cell.get(direction, True):
                return directions[direction]

        return 0.0
