"""Maze generation using recursive backtracking algorithm.

Based on examples/maze_generator.py by KS 2022.
"""

import random
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


class MazeGenerator:
    """Generate a maze using recursive backtracking (DFS).

    Based on the maze_generator.py example.
    """

    def __init__(self, rows: int, cols: int, seed: Optional[int] = None):
        """Initialize maze generator.

        Args:
            rows: Number of cell rows
            cols: Number of cell columns
            seed: Random seed for reproducibility
        """
        self.rows = rows
        self.cols = cols
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Each cell has 4 walls: N, S, E, W
        # True = wall exists, False = wall removed (passage)
        self.grid: List[List[dict]] = []

        # Walls array similar to maze_generator.py example
        # walls[y, x, 0] = visited status (0=visited, 1=not visited, -1=edge)
        # walls[y, x, 1] = down wall (S)
        # walls[y, x, 2] = right wall (E)
        # Add +2 to dimensions for border cells
        self.walls: NDArray[np.int8] = np.ones((rows + 2, cols + 2, 3), dtype=np.int8)
        # Mark edges as unusable (-1)
        self.walls[:, 0, 0] = -1
        self.walls[:, cols + 1, 0] = -1
        self.walls[0, :, 0] = -1
        self.walls[rows + 1, :, 0] = -1

    def generate(self) -> List[List[dict]]:
        """Generate maze using recursive backtracking.

        Returns:
            Grid of cells, each with wall flags {N, S, E, W}
        """
        # Reset walls array
        self.walls[1:-1, 1:-1, :] = 1

        # Direction vectors: up, down, left, right (row, col)
        up = np.array([-1, 0], dtype=np.int64)
        down = np.array([1, 0], dtype=np.int64)
        left = np.array([0, -1], dtype=np.int64)
        right = np.array([0, 1], dtype=np.int64)

        # Start from a random cell (using 1-indexed for the walls array)
        cell: NDArray[np.int64] = np.array(
            [random.randrange(1, self.rows + 1), random.randrange(1, self.cols + 1)],
            dtype=np.int64,
        )
        self.walls[cell[0], cell[1], 0] = 0  # Mark as visited

        need_cell_range = False
        corridor_len = 999  # No limit on corridor length
        corridor_start = 0
        round_nr = 0

        while np.size(cell) > 0:
            round_nr += 1

            # Get the four neighbors for current cell
            cell_neighbors: NDArray[np.int64] = np.vstack(
                (cell + up, cell + left, cell + down, cell + right)
            ).astype(np.int64)

            # Valid neighbors are unvisited (status == 1)
            # Use integer array indexing
            row_idx = cell_neighbors[:, 0]
            col_idx = cell_neighbors[:, 1]
            visited_status = self.walls[row_idx, col_idx, 0]
            valid_neighbors = cell_neighbors[visited_status == 1]

            if np.size(valid_neighbors) > 0:
                # There is at least one valid neighbor, pick one at random
                neighbor: NDArray[np.int64] = valid_neighbors[
                    random.randrange(0, np.shape(valid_neighbors)[0]), :
                ].astype(np.int64)

                if np.size(cell) > 2:
                    # If cell is an array of cells, pick one cell adjacent to neighbor
                    cell = cell[np.sum(abs(cell - neighbor), axis=1) == 1]
                    cell = cell[random.randrange(0, np.shape(cell)[0]), :].astype(
                        np.int64
                    )

                # Mark neighbor as visited
                self.walls[neighbor[0], neighbor[1], 0] = 0

                # Remove wall between current cell and neighbor
                # Wall is stored in the cell with smaller coordinates
                # walls[y, x, 1] = down wall, walls[y, x, 2] = right wall
                min_y = int(min(cell[0], neighbor[0]))
                min_x = int(min(cell[1], neighbor[1]))
                # 1 + abs(neighbor[1] - cell[1]): if horizontal move (x differs), use index 2 (right wall)
                # if vertical move (y differs), use index 1 (down wall)
                wall_idx = 1 + int(abs(neighbor[1] - cell[1]))
                self.walls[min_y, min_x, wall_idx] = 0

                # Check if more corridor length is available
                if round_nr - corridor_start < corridor_len:
                    # Continue current corridor
                    cell = np.array([neighbor[0], neighbor[1]], dtype=np.int64)
                else:
                    # Maximum corridor length reached, start new junction
                    need_cell_range = True
            else:
                # No valid neighbors for this cell
                if np.size(cell) > 2:
                    # Cell already contains array of cells, no more valid neighbors exist
                    cell = np.zeros((0, 0), dtype=np.int64)  # End the loop
                else:
                    # Dead end, start new junction
                    need_cell_range = True

            if need_cell_range:
                # Get all visited cells (=0) not marked as "no neighbors" (=-1)
                cell = (
                    np.transpose(np.nonzero(self.walls[1:-1, 1:-1, 0] == 0)) + 1
                ).astype(np.int64)

                if np.size(cell) == 0:
                    break

                # Check these for valid neighbors
                cell_rows = cell[:, 0]
                cell_cols = cell[:, 1]
                valid_neighbor_exists = np.array(
                    [
                        self.walls[cell_rows - 1, cell_cols, 0],
                        self.walls[cell_rows + 1, cell_cols, 0],
                        self.walls[cell_rows, cell_cols - 1, 0],
                        self.walls[cell_rows, cell_cols + 1, 0],
                    ]
                ).max(axis=0)

                # Get all visited cells with no valid neighbors
                cell_no_neighbors = cell[valid_neighbor_exists != 1]
                # Mark these so they won't be used again
                if len(cell_no_neighbors) > 0:
                    no_neighbor_rows = cell_no_neighbors[:, 0]
                    no_neighbor_cols = cell_no_neighbors[:, 1]
                    self.walls[no_neighbor_rows, no_neighbor_cols, 0] = -1

                corridor_start = round_nr
                need_cell_range = False

        # Convert walls array to grid format
        return self._walls_to_grid()

    def _walls_to_grid(self) -> List[List[dict]]:
        """Convert internal walls representation to grid of wall dictionaries.

        The walls array stores:
        - walls[y, x, 1] = down wall (between cell (y,x) and cell (y+1,x))
        - walls[y, x, 2] = right wall (between cell (y,x) and cell (y,x+1))

        We need to convert to:
        - N wall of cell (row, col) = down wall of cell (row-1, col) in walls array
        - S wall of cell (row, col) = down wall of cell (row, col) in walls array
        - W wall of cell (row, col) = right wall of cell (row, col-1) in walls array
        - E wall of cell (row, col) = right wall of cell (row, col) in walls array
        """
        self.grid = []

        for row in range(self.rows):
            grid_row = []
            for col in range(self.cols):
                # walls array is 1-indexed (due to border)
                wy = row + 1
                wx = col + 1

                cell_walls = {
                    "N": True,  # Top wall
                    "S": True,  # Bottom wall
                    "E": True,  # Right wall
                    "W": True,  # Left wall
                }

                # N wall: check down wall of cell above (wy-1, wx)
                # But for row 0, there's no cell above, so N is always True (outer wall)
                if row > 0:
                    # Down wall of cell above us
                    if self.walls[wy - 1, wx, 1] == 0:
                        cell_walls["N"] = False
                # else: row 0, N wall is outer boundary, always True

                # S wall: check down wall of current cell (wy, wx)
                # But for last row, S is always True (outer wall)
                if row < self.rows - 1:
                    if self.walls[wy, wx, 1] == 0:
                        cell_walls["S"] = False
                # else: last row, S wall is outer boundary, always True

                # W wall: check right wall of cell to the left (wy, wx-1)
                # But for col 0, W is always True (outer wall)
                if col > 0:
                    if self.walls[wy, wx - 1, 2] == 0:
                        cell_walls["W"] = False
                # else: col 0, W wall is outer boundary, always True

                # E wall: check right wall of current cell (wy, wx)
                # But for last col, E is always True (outer wall)
                if col < self.cols - 1:
                    if self.walls[wy, wx, 2] == 0:
                        cell_walls["E"] = False
                # else: last col, E wall is outer boundary, always True

                grid_row.append(cell_walls)
            self.grid.append(grid_row)

        return self.grid

    def get_cell_walls(self, row: int, col: int) -> dict:
        """Get wall configuration for a cell.

        Args:
            row: Cell row
            col: Cell column

        Returns:
            Dict with N, S, E, W wall flags
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return {"N": True, "S": True, "E": True, "W": True}

    def get_2d_blocks(self) -> NDArray[np.int8]:
        """Get maze as 2D block array (like maze_generator.py example).

        Returns:
            2D numpy array where 1=wall, 0=passage
        """
        block_rows = self.rows * 2 + 1
        block_cols = self.cols * 2 + 1
        blocks: NDArray[np.int8] = np.ones((block_rows, block_cols), dtype=np.int8)

        # Every cell center is a passage (odd indices)
        blocks[1::2, 1::2] = 0

        # Horizontal passages (between vertically adjacent cells)
        # walls[y, x, 1] = 0 means passage between (y,x) and (y+1,x)
        # In blocks: row = y*2+2, col = x*2+1
        for y in range(1, self.rows + 1):
            for x in range(1, self.cols + 1):
                if self.walls[y, x, 1] == 0:  # Down wall removed
                    blocks[y * 2, x * 2 - 1] = 0

        # Vertical passages (between horizontally adjacent cells)
        # walls[y, x, 2] = 0 means passage between (y,x) and (y,x+1)
        # In blocks: row = y*2-1, col = x*2
        for y in range(1, self.rows + 1):
            for x in range(1, self.cols + 1):
                if self.walls[y, x, 2] == 0:  # Right wall removed
                    blocks[y * 2 - 1, x * 2] = 0

        return blocks


def generate_maze(rows: int, cols: int, seed: Optional[int] = None) -> List[List[dict]]:
    """Generate a maze grid.

    Args:
        rows: Number of rows
        cols: Number of columns
        seed: Random seed

    Returns:
        Grid of wall configurations
    """
    generator = MazeGenerator(rows, cols, seed)
    return generator.generate()
