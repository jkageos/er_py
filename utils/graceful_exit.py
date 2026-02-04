"""Graceful exit handling for long-running evolutionary processes."""

import signal
import sys
from typing import Any, Callable, Optional


class GracefulExitHandler:
    """Handle graceful shutdown on Ctrl+C (SIGINT) and SIGTERM.

    Usage:
        handler = GracefulExitHandler()

        for generation in range(max_generations):
            if handler.should_exit:
                break
            # ... evolution step ...

        if handler.exit_requested:
            print("Evolution interrupted by user")
    """

    def __init__(
        self,
        on_exit: Optional[Callable[[], None]] = None,
        message: str = "\n\nGraceful shutdown requested. Finishing current generation...",
    ):
        """Initialize graceful exit handler.

        Args:
            on_exit: Optional callback to run when exit is requested
            message: Message to display when exit is requested
        """
        self._exit_requested = False
        self._exit_count = 0
        self._on_exit = on_exit
        self._message = message
        # Use Any for signal handlers since they can be callable, int, or None
        self._original_sigint: Any = None
        self._original_sigterm: Any = None
        self._registered = False

    @property
    def should_exit(self) -> bool:
        """Check if exit has been requested."""
        return self._exit_requested

    @property
    def exit_requested(self) -> bool:
        """Alias for should_exit."""
        return self._exit_requested

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle interrupt signals."""
        self._exit_count += 1

        if self._exit_count == 1:
            # First interrupt: request graceful exit
            print(self._message)
            self._exit_requested = True

            if self._on_exit:
                try:
                    self._on_exit()
                except Exception:
                    pass  # Don't let callback errors prevent shutdown
        elif self._exit_count == 2:
            # Second interrupt: warn about force exit
            print("\nPress Ctrl+C once more to force quit immediately...")
        else:
            # Third interrupt: force exit
            print("\nForce exit requested. Terminating immediately...")
            sys.exit(1)

    def register(self) -> "GracefulExitHandler":
        """Register signal handlers.

        Returns:
            Self for method chaining
        """
        if not self._registered:
            self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
            # SIGTERM might not be available on all platforms
            try:
                self._original_sigterm = signal.signal(
                    signal.SIGTERM, self._signal_handler
                )
            except (AttributeError, ValueError):
                pass
            self._registered = True
        return self

    def unregister(self) -> None:
        """Restore original signal handlers."""
        if self._registered:
            if self._original_sigint is not None:
                try:
                    signal.signal(signal.SIGINT, self._original_sigint)
                except (ValueError, OSError):
                    pass  # May fail if called from wrong thread
            if self._original_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, self._original_sigterm)
                except (AttributeError, ValueError, OSError):
                    pass
            self._registered = False

    def reset(self) -> None:
        """Reset exit state (useful for multiple runs)."""
        self._exit_requested = False
        self._exit_count = 0

    def __enter__(self) -> "GracefulExitHandler":
        """Context manager entry."""
        return self.register()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit."""
        self.unregister()
        return False  # Don't suppress exceptions


class EvolutionInterrupted(Exception):
    """Exception raised when evolution is interrupted by user."""

    def __init__(self, generation: int, message: str = "Evolution interrupted by user"):
        self.generation = generation
        super().__init__(f"{message} at generation {generation}")
