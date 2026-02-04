"""Video and trajectory recording for evolved controllers."""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Suppress pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


class TrajectoryRecorder:
    """Record and save trajectory data."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trajectories: List[List[Tuple[float, float]]] = []
        self.metadata: List[dict] = []

    def record(
        self,
        trajectory: List[Tuple[float, float]],
        goal_reached: bool = False,
        novelty_score: float = 0.0,
        generation: int = 0,
        genome_id: int = 0,
    ) -> None:
        """Record a trajectory with metadata."""
        self.trajectories.append(trajectory)
        self.metadata.append(
            {
                "goal_reached": goal_reached,
                "novelty_score": novelty_score,
                "generation": generation,
                "genome_id": genome_id,
                "path_length": self._compute_path_length(trajectory),
                "final_position": trajectory[-1] if trajectory else (0, 0),
            }
        )

    def _compute_path_length(self, trajectory: List[Tuple[float, float]]) -> float:
        """Compute total path length."""
        if len(trajectory) < 2:
            return 0.0

        import math

        total = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def save(self, filename: str = "trajectories.pkl") -> Path:
        """Save recorded trajectories."""
        filepath = self.output_dir / filename
        data = {
            "trajectories": self.trajectories,
            "metadata": self.metadata,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return filepath

    def load(self, filename: str = "trajectories.pkl") -> None:
        """Load recorded trajectories."""
        filepath = self.output_dir / filename
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.trajectories = data["trajectories"]
        self.metadata = data["metadata"]

    def _novelty_key(self, i: int) -> float:
        """Key function for sorting by novelty."""
        return self.metadata[i]["novelty_score"]

    def _path_length_key(self, i: int) -> float:
        """Key function for sorting by path length."""
        return self.metadata[i]["path_length"]

    def _goal_key(self, i: int) -> Tuple[bool, float]:
        """Key function for sorting by goal (prioritize goal-reaching, then novelty)."""
        return (self.metadata[i]["goal_reached"], self.metadata[i]["novelty_score"])

    def get_best_trajectories(self, n: int = 5, by: str = "novelty") -> List[int]:
        """Get indices of best trajectories by criterion."""
        if by == "novelty":
            key_func = self._novelty_key
        elif by == "path_length":
            key_func = self._path_length_key
        elif by == "goal":
            # Use a wrapper that returns a comparable tuple
            indices = sorted(
                range(len(self.trajectories)), key=self._goal_key, reverse=True
            )
            return indices[:n]
        else:
            raise ValueError(f"Unknown criterion: {by}")

        indices = sorted(range(len(self.trajectories)), key=key_func, reverse=True)
        return indices[:n]


class VideoRecorder:
    """Record video of robot evaluation."""

    def __init__(
        self,
        output_dir: str = "results",
        fps: int = 60,
        width: int = 1200,
        height: int = 1000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height

        self._frames: List[np.ndarray] = []
        self._recording = False

    def start_recording(self) -> None:
        """Start recording frames."""
        self._frames = []
        self._recording = True

    def capture_frame(self, surface) -> None:
        """Capture a frame from pygame surface."""
        if not self._recording:
            return

        try:
            import pygame

            # Convert surface to numpy array
            frame = pygame.surfarray.array3d(surface)
            # Transpose from (width, height, 3) to (height, width, 3)
            frame = np.transpose(frame, (1, 0, 2))
            self._frames.append(frame)
        except Exception as e:
            print(f"Warning: Could not capture frame: {e}")

    def stop_recording(self) -> None:
        """Stop recording."""
        self._recording = False

    def save_video(self, filename: str = "controller_video.mp4") -> Optional[Path]:
        """Save recorded frames as video."""
        if not self._frames:
            print("No frames to save")
            return None

        filepath = self.output_dir / filename

        try:
            import cv2  # type: ignore[import-not-found]

            # Get frame dimensions from first frame
            height, width = self._frames[0].shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))

            for frame in self._frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"Saved video to {filepath}")
            return filepath

        except ImportError:
            print("Warning: OpenCV not installed. Saving frames as images instead.")
            return self._save_as_images(filename.replace(".mp4", ""))

    def _save_as_images(self, prefix: str) -> Path:
        """Fallback: save frames as individual images."""
        frames_dir = self.output_dir / f"{prefix}_frames"
        frames_dir.mkdir(exist_ok=True)

        try:
            from PIL import Image

            for i, frame in enumerate(self._frames):
                img = Image.fromarray(frame)
                img.save(frames_dir / f"frame_{i:05d}.png")

            print(f"Saved {len(self._frames)} frames to {frames_dir}")
            return frames_dir

        except ImportError:
            # Save as numpy arrays
            np.save(frames_dir / "frames.npy", np.array(self._frames))
            print(f"Saved frames as numpy array to {frames_dir}")
            return frames_dir

    def save_gif(
        self, filename: str = "controller.gif", duration: int = 50
    ) -> Optional[Path]:
        """Save recorded frames as GIF."""
        if not self._frames:
            print("No frames to save")
            return None

        filepath = self.output_dir / filename

        try:
            from PIL import Image

            # Convert frames to PIL images
            images = [Image.fromarray(frame) for frame in self._frames]

            # Save as GIF (use every nth frame for smaller file)
            skip = max(1, len(images) // 200)  # Target ~200 frames max
            images_subset = images[::skip]

            images_subset[0].save(
                filepath,
                save_all=True,
                append_images=images_subset[1:],
                duration=duration * skip,
                loop=0,
            )

            print(f"Saved GIF to {filepath}")
            return filepath

        except ImportError:
            print("Warning: PIL not installed. Cannot save GIF.")
            return None


def record_controller_run(
    genome: np.ndarray,
    config,
    output_dir: str = "results",
    save_video: bool = True,
    save_gif: bool = True,
) -> Tuple[List[Tuple[float, float]], bool, Optional[Path], Optional[Path]]:
    """Record a full evaluation run of a controller.

    Args:
        genome: Controller genome
        config: Task11Config
        output_dir: Output directory
        save_video: Whether to save MP4 video
        save_gif: Whether to save GIF

    Returns:
        (trajectory, goal_reached, video_path, gif_path)
    """
    from task11.controller import MazeController
    from task11.environment import MazeEnvironment

    # Create environment with rendering
    env = MazeEnvironment(
        config=config.environment,
        maze_file=config.maze_file,
        render=True,
        spawn_robot=True,
    )

    controller = MazeController(
        n_hidden=config.evolution.n_hidden,
        n_hidden_layers=config.evolution.n_hidden_layers,
        genome=genome,
    )

    env.reset(controller=controller)

    # Setup recorders
    video_recorder = VideoRecorder(
        output_dir=output_dir,
        fps=60,
        width=config.environment.width,
        height=config.environment.height,
    )

    if save_video or save_gif:
        video_recorder.start_recording()

    # Run evaluation
    trajectory: List[Tuple[float, float]] = []
    for step in range(config.environment.eval_steps):
        if not env.step():
            break

        env.render()
        trajectory.append(env.get_position())

        if env.screen is not None and (save_video or save_gif):
            video_recorder.capture_frame(env.screen)

        if env.screen is None:
            break

    video_recorder.stop_recording()
    goal_reached = env.is_goal_reached()

    # Save recordings
    video_path: Optional[Path] = None
    gif_path: Optional[Path] = None

    if save_video:
        video_path = video_recorder.save_video("best_controller.mp4")

    if save_gif:
        gif_path = video_recorder.save_gif("best_controller.gif")

    env.close()

    return trajectory, goal_reached, video_path, gif_path
