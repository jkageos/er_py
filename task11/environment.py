"""Environment factory and utilities."""

import math
import pickle
from pathlib import Path
from typing import List, Tuple

import pymunk

from task11.config import EnvironmentConfig, get_config
from task11.controller import MazeController
from task11.maze_generator import generate_maze
from task11.robot_agent import RobotAgent


class MazeEnvironment:
    """2D maze environment with physics simulation."""

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        maze_file: str | Path | None = None,
        render: bool = False,
        spawn_robot: bool = True,
    ):
        """Initialize maze environment.

        Args:
            config: Environment configuration (uses default if None)
            maze_file: Path to maze file (load/save)
            render: Enable rendering
            spawn_robot: Whether to spawn robot on init
        """
        self.config = config or get_config().environment
        self.render_enabled = render
        self.maze_file = Path(maze_file) if maze_file else None

        # Derived values
        self.width = self.config.width
        self.height = self.config.height
        self.robot_radius = self.config.robot_radius
        self.max_sensor_range = self.config.max_sensor_range
        self.max_speed = self.config.max_speed
        self.wall_thickness = self.config.wall_thickness
        self.maze_cols = self.config.maze_cols
        self.maze_rows = self.config.maze_rows

        # Calculate cell dimensions
        margin = self.wall_thickness
        self.cell_width = (self.width - 2 * margin) / self.maze_cols
        self.cell_height = (self.height - 2 * margin) / self.maze_rows

        # Load or generate maze
        self.maze_grid = self._load_or_generate_maze()

        # Start and goal positions
        self.start_pos = (
            margin + self.cell_width / 2,
            margin + self.cell_height / 2,
        )
        self.goal_pos = (
            self.width - margin - self.cell_width / 2,
            self.height - margin - self.cell_height / 2,
        )

        # Goal radius for detection (robot touches goal zone)
        self.goal_radius = min(self.cell_width, self.cell_height) / 3

        # Track if goal was reached
        self._goal_reached = False

        # Physics setup
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = self.config.physics.damping
        self.space.iterations = self.config.physics.iterations

        # Robot
        self.robot: RobotAgent | None = None

        # Rendering - import pygame types for proper typing
        try:
            import pygame

            self.screen: pygame.Surface | None = None
            self.clock: pygame.time.Clock | None = None
        except ImportError:
            self.screen = None
            self.clock = None

        if self.render_enabled:
            self._init_rendering()

        # Create walls
        self._create_walls()

        # Compute initial angle
        self._initial_angle = RobotAgent.compute_initial_angle(
            self.maze_grid, robot_row=0, robot_col=0
        )

        if spawn_robot:
            self.reset()

    def _load_or_generate_maze(self) -> List[List[dict]]:
        """Load maze from file or generate new one."""
        if self.maze_file and self.maze_file.exists():
            with open(self.maze_file, "rb") as f:
                grid = pickle.load(f)
            # Validate dimensions
            if len(grid) == self.maze_rows and len(grid[0]) == self.maze_cols:
                return grid

        # Generate new maze
        grid = generate_maze(self.maze_rows, self.maze_cols, self.config.seed)

        # Save if path provided
        if self.maze_file:
            self.maze_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.maze_file, "wb") as f:
                pickle.dump(grid, f)

        return grid

    def _init_rendering(self) -> None:
        """Initialize pygame rendering."""
        import pygame

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Maze Environment")
        self.clock = pygame.time.Clock()

    def _create_walls(self) -> None:
        """Create all maze walls as physics bodies."""
        margin = self.wall_thickness

        # Outer boundary
        self._add_wall_rect(0, 0, self.width, self.wall_thickness)
        self._add_wall_rect(
            0, self.height - self.wall_thickness, self.width, self.wall_thickness
        )
        self._add_wall_rect(0, 0, self.wall_thickness, self.height)
        self._add_wall_rect(
            self.width - self.wall_thickness, 0, self.wall_thickness, self.height
        )

        # Internal walls
        for row in range(self.maze_rows):
            for col in range(self.maze_cols):
                cell = self.maze_grid[row][col]
                cell_x = margin + col * self.cell_width
                cell_y = margin + row * self.cell_height

                # South wall
                if cell.get("S", False) and row < self.maze_rows - 1:
                    self._add_wall_rect(
                        cell_x,
                        cell_y + self.cell_height - self.wall_thickness / 2,
                        self.cell_width,
                        self.wall_thickness,
                    )

                # East wall
                if cell.get("E", False) and col < self.maze_cols - 1:
                    self._add_wall_rect(
                        cell_x + self.cell_width - self.wall_thickness / 2,
                        cell_y,
                        self.wall_thickness,
                        self.cell_height,
                    )

    def _add_wall_rect(self, x: float, y: float, w: float, h: float) -> None:
        """Add a rectangular wall."""
        half_w, half_h = w / 2, h / 2
        center_x, center_y = x + half_w, y + half_h

        vertices = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        shape = pymunk.Poly(
            self.space.static_body, vertices, pymunk.Transform(tx=center_x, ty=center_y)
        )
        shape.friction = 1.0
        shape.elasticity = 0.0
        shape.collision_type = 2
        self.space.add(shape)

    def reset(self, controller: MazeController | None = None) -> None:
        """Reset environment with optional new controller."""
        if self.robot is not None:
            self.space.remove(self.robot.body, self.robot.shape)

        self._goal_reached = False

        self.robot = RobotAgent(
            x=self.start_pos[0],
            y=self.start_pos[1],
            robot_radius=self.robot_radius,
            max_sensor_range=self.max_sensor_range,
            max_speed=self.max_speed,
            controller=controller,
            physics=vars(self.config.physics),
            lidar_config=vars(self.config.lidar),
            initial_angle=self._initial_angle,
        )
        self.space.add(self.robot.body, self.robot.shape)

    def step(self) -> bool:
        """Step simulation forward.

        Returns:
            True if episode should continue, False if goal reached
        """
        if self.robot and not self._goal_reached:
            self.robot.update(self.space)

            # Check if robot reached goal
            if self._check_goal_reached():
                self._goal_reached = True
                return False

        return not self._goal_reached

    def _check_goal_reached(self) -> bool:
        """Check if robot has reached the goal zone."""
        if self.robot is None:
            return False

        pos = self.robot.body.position
        dx = pos.x - self.goal_pos[0]
        dy = pos.y - self.goal_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Goal reached if robot center is within goal_radius + robot_radius
        return distance <= (self.goal_radius + self.robot_radius)

    def is_goal_reached(self) -> bool:
        """Check if goal was reached."""
        return self._goal_reached

    def get_position(self) -> Tuple[float, float]:
        """Get robot position."""
        return self.robot.get_position() if self.robot else self.start_pos

    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get trajectory history."""
        return self.robot.get_trajectory() if self.robot else []

    def render(self) -> None:
        """Render the environment."""
        if not self.render_enabled or self.screen is None:
            return

        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        self._draw_maze()
        self._draw_markers()

        if self.robot:
            self._draw_robot()

        pygame.display.flip()
        if self.clock:
            self.clock.tick(60)

    def _draw_maze(self) -> None:
        """Draw maze walls."""
        if self.screen is None:
            return

        import pygame

        wall_color = (40, 40, 40)
        margin = self.wall_thickness

        # Outer boundary
        pygame.draw.rect(
            self.screen, wall_color, (0, 0, self.width, self.wall_thickness)
        )
        pygame.draw.rect(
            self.screen,
            wall_color,
            (0, self.height - self.wall_thickness, self.width, self.wall_thickness),
        )
        pygame.draw.rect(
            self.screen, wall_color, (0, 0, self.wall_thickness, self.height)
        )
        pygame.draw.rect(
            self.screen,
            wall_color,
            (self.width - self.wall_thickness, 0, self.wall_thickness, self.height),
        )

        # Internal walls
        for row in range(self.maze_rows):
            for col in range(self.maze_cols):
                cell = self.maze_grid[row][col]
                cell_x = margin + col * self.cell_width
                cell_y = margin + row * self.cell_height

                if cell.get("S", False) and row < self.maze_rows - 1:
                    pygame.draw.rect(
                        self.screen,
                        wall_color,
                        (
                            cell_x,
                            cell_y + self.cell_height - self.wall_thickness / 2,
                            self.cell_width,
                            self.wall_thickness,
                        ),
                    )

                if cell.get("E", False) and col < self.maze_cols - 1:
                    pygame.draw.rect(
                        self.screen,
                        wall_color,
                        (
                            cell_x + self.cell_width - self.wall_thickness / 2,
                            cell_y,
                            self.wall_thickness,
                            self.cell_height,
                        ),
                    )

    def _draw_markers(self) -> None:
        """Draw start and goal markers."""
        if self.screen is None:
            return

        import pygame

        start_radius = max(8, int(self.robot_radius * 0.6))

        # Draw goal zone (filled circle showing detection area)
        goal_color = (255, 200, 200) if not self._goal_reached else (200, 255, 200)
        pygame.draw.circle(
            self.screen,
            goal_color,
            (int(self.goal_pos[0]), int(self.goal_pos[1])),
            int(self.goal_radius + self.robot_radius),
        )

        # Start marker
        pygame.draw.circle(
            self.screen,
            (0, 200, 0),
            (int(self.start_pos[0]), int(self.start_pos[1])),
            start_radius + 3,
            3,
        )

        # Goal marker (outline)
        goal_outline_color = (0, 200, 0) if self._goal_reached else (200, 0, 0)
        pygame.draw.circle(
            self.screen,
            goal_outline_color,
            (int(self.goal_pos[0]), int(self.goal_pos[1])),
            int(self.goal_radius),
            3,
        )

        font = pygame.font.Font(None, 20)
        self.screen.blit(
            font.render("S", True, (0, 150, 0)),
            (int(self.start_pos[0]) - start_radius - 15, int(self.start_pos[1]) - 8),
        )
        self.screen.blit(
            font.render("G", True, (150, 0, 0)),
            (
                int(self.goal_pos[0]) + int(self.goal_radius) + 5,
                int(self.goal_pos[1]) - 8,
            ),
        )

    def _draw_robot(self) -> None:
        """Draw the robot."""
        if self.screen is None or self.robot is None:
            return

        import math

        import pygame

        pos = self.robot.body.position
        angle = self.robot.body.angle

        # Robot body
        pygame.draw.circle(
            self.screen,
            (100, 100, 200),
            (int(pos.x), int(pos.y)),
            int(self.robot_radius),
        )

        # Direction indicator
        end_x = pos.x + math.cos(angle) * self.robot_radius
        end_y = pos.y + math.sin(angle) * self.robot_radius
        pygame.draw.line(
            self.screen,
            (255, 0, 0),
            (int(pos.x), int(pos.y)),
            (int(end_x), int(end_y)),
            2,
        )

        # LiDAR rays
        for start, end in self.robot.lidar.get_ray_endpoints(
            self.robot.body, self.robot_radius
        ):
            pygame.draw.line(
                self.screen,
                (0, 255, 0),
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                1,
            )

    def close(self) -> None:
        """Clean up resources."""
        if self.render_enabled:
            import pygame

            pygame.quit()
        self.screen = None
