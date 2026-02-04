"""Evolution logging utilities."""

import time
from pathlib import Path
from typing import Optional


class EvolutionLogger:
    """Log evolution progress and statistics."""

    def __init__(self, log_file: Optional[str] = None, print_every: int = 5):
        """
        Initialize logger.

        Args:
            log_file: Optional log file path
            print_every: Print progress every N generations
        """
        self.log_file = Path(log_file) if log_file else None
        self.print_every = print_every
        self.start_time = None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                f.write("Evolution Log\n")
                f.write("=" * 80 + "\n\n")

    def start(self, task_name: str, config: dict):
        """Log evolution start."""
        self.start_time = time.time()
        msg = f"Starting {task_name}\n"
        msg += f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        for key, value in config.items():
            msg += f"  {key}: {value}\n"
        msg += "\n"
        self._write(msg)
        print(msg)

    def _write(self, message: str):
        """Write to log file."""
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message)
