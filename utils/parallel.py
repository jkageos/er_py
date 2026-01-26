"""Parallel processing utilities for evolutionary algorithms."""

import multiprocessing as mp
from typing import Any, Callable, List, Optional

__all__ = ["parallel_map", "get_optimal_workers"]


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

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(func, items, chunksize=chunksize)

    return results
