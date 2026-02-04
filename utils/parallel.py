"""Parallel processing utilities for evolutionary algorithms."""

import multiprocessing as mp
import signal
from typing import Any, Callable, List, Optional

__all__ = ["parallel_map", "get_optimal_workers"]


def _init_worker() -> None:
    """Initialize worker process to ignore SIGINT.

    This ensures Ctrl+C is only handled by the main process,
    allowing for graceful shutdown.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_optimal_workers(max_workers: Optional[int] = None) -> int:
    """
    Get optimal number of worker processes.

    Args:
        max_workers: Maximum number of workers (None for auto-detect)

    Returns:
        Number of workers to use
    """
    cpu_count = mp.cpu_count()

    if max_workers is None:
        # Use all CPUs minus 2 to keep system responsive and safe
        return max(1, cpu_count - 2)
    else:
        # Ensure we never exceed cpu_count - 2 for safety
        return max(1, min(max_workers, cpu_count - 2))


def parallel_map(
    func: Callable[[Any], Any],
    items: List[Any],
    n_workers: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> List[Any]:
    """
    Parallel map function using multiprocessing.

    Args:
        func: Function to apply to each item
        items: List of items to process
        n_workers: Number of worker processes (None for auto-detect)
        chunksize: Chunk size for parallel processing (None for auto)

    Returns:
        List of results in the same order as items
    """
    if not items:
        return []

    n_workers = get_optimal_workers(n_workers)

    # For small workloads, don't bother with parallelization
    if len(items) < n_workers * 2:
        return [func(item) for item in items]

    # Auto-calculate chunksize if not specified
    if chunksize is None:
        chunksize = max(1, len(items) // (n_workers * 4))

    # Use initializer to make workers ignore SIGINT
    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        try:
            results = pool.map(func, items, chunksize=chunksize)
        except KeyboardInterrupt:
            # If interrupted, terminate workers and re-raise
            pool.terminate()
            pool.join()
            raise

    return results
