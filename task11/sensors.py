"""Sensor systems for robot: LiDAR and Bumpers."""

import math
from typing import List, Tuple

import numpy as np
import pymunk


class LiDAR:
    """Configurable LiDAR sensor array."""

    def __init__(
        self,
        max_range: float = 100.0,
        num_rays: int = 5,
    ):
        self.max_range = max_range
        self.num_rays = num_rays

        # Distribute rays evenly in front hemisphere
        if num_rays == 3:
            self.angles = np.array([-45.0, 0.0, 45.0])
        elif num_rays == 5:
            # Better coverage: -90, -45, 0, 45, 90 degrees
            self.angles = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
        elif num_rays == 7:
            self.angles = np.linspace(-90, 90, 7)
        else:
            # Spread rays over 180 degrees in front
            self.angles = np.linspace(-90, 90, num_rays)

        self.readings: np.ndarray = np.zeros(num_rays, dtype=np.float64)
        self._raw_distances: np.ndarray = np.full(num_rays, max_range, dtype=np.float64)

    def update(
        self,
        robot_body: pymunk.Body,
        space: pymunk.Space,
        robot_shape: pymunk.Shape,
        robot_radius: float,
    ) -> None:
        """Cast rays and update sensor readings.

        Readings are normalized to [0, 1] where:
        - 0 = obstacle very close (danger!)
        - 1 = no obstacle detected (safe to move)

        This convention matches Kheperax where higher values mean more space.
        """
        # Filter to ignore robot's own shape
        shape_filter = pymunk.ShapeFilter(group=robot_shape.filter.group)

        # Pre-compute angles in radians
        angles_rad = np.radians(self.angles) + robot_body.angle

        for i, sensor_angle in enumerate(angles_rad):
            direction = pymunk.Vec2d(math.cos(sensor_angle), math.sin(sensor_angle))

            # Start ray at robot surface
            start = robot_body.position + direction * (robot_radius + 1)
            end = robot_body.position + direction * self.max_range

            # Perform raycast
            query = space.segment_query_first(start, end, 1, shape_filter)

            if query and query.shape:
                # Calculate distance to hit point
                hit_point = query.point
                distance = (hit_point - robot_body.position).length - robot_radius
                self._raw_distances[i] = max(0.0, distance)
                # Normalize: farther = higher reading (more space = safer)
                self.readings[i] = min(1.0, distance / self.max_range)
            else:
                # No hit = max range = safe
                self._raw_distances[i] = self.max_range
                self.readings[i] = 1.0

    def get_readings(self) -> List[float]:
        """Get sensor readings."""
        return self.readings.tolist()

    def get_ray_endpoints(
        self,
        robot_body: pymunk.Body,
        robot_radius: float,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get ray endpoints for visualization."""
        rays = []
        angles_rad = np.radians(self.angles) + robot_body.angle

        for sensor_angle, raw_dist in zip(angles_rad, self._raw_distances):
            direction = pymunk.Vec2d(math.cos(sensor_angle), math.sin(sensor_angle))
            start = robot_body.position + direction * robot_radius
            end = start + direction * raw_dist
            rays.append(((start.x, start.y), (end.x, end.y)))

        return rays


class BumperSensors:
    """Bumper sensors for collision detection.

    Two bumpers: left and right, covering front hemisphere.
    Returns 1.0 when touching a wall, 0.0 otherwise.
    """

    def __init__(self):
        self.left_bumper: float = 0.0
        self.right_bumper: float = 0.0
        self._collision_points: List[Tuple[float, float]] = []

    def update(
        self,
        robot_body: pymunk.Body,
        robot_shape: pymunk.Shape,
        space: pymunk.Space,
    ) -> None:
        """Update bumper readings based on current collisions."""
        self.left_bumper = 0.0
        self.right_bumper = 0.0
        self._collision_points = []

        # Query shapes near the robot
        robot_pos = robot_body.position
        robot_angle = robot_body.angle

        # Check all contact points
        for arbiter in space.shape_query(robot_shape):
            if arbiter.shape is not robot_shape:
                # Get contact points
                contact_set = arbiter.contact_point_set
                for point in contact_set.points:
                    contact_pos = point.point_a

                    # Determine if contact is on left or right side
                    to_contact = contact_pos - robot_pos
                    forward = pymunk.Vec2d(math.cos(robot_angle), math.sin(robot_angle))
                    cross = forward.cross(to_contact)
                    dot = forward.dot(to_contact)

                    if dot > 0:  # Contact is in front hemisphere
                        self._collision_points.append((contact_pos.x, contact_pos.y))
                        if cross > 0:
                            self.left_bumper = 1.0
                        else:
                            self.right_bumper = 1.0

    def get_readings(self) -> List[float]:
        """Get bumper readings [left, right]."""
        return [self.left_bumper, self.right_bumper]

    def is_colliding(self) -> bool:
        """Check if any bumper is active."""
        return self.left_bumper > 0 or self.right_bumper > 0

    def get_collision_points(self) -> List[Tuple[float, float]]:
        """Get collision points for visualization."""
        return self._collision_points.copy()
