"""Read aligned sensors on a common grid without a large in-memory join."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq


class SensorCursor:
    """A small batch per sensor replaces a large, RAM-intensive wide join."""

    def __init__(self, path):
        self.source = pq.ParquetFile(path)
        self.iterator = self.source.iter_batches(batch_size=4096)
        self._advance()

    def _advance(self):
        self.batch = next(self.iterator, None)
        while self.batch is not None and self.batch.num_rows == 0:
            self.batch = next(self.iterator, None)
        if self.batch is not None:
            self.ticks = self.batch.column(0).to_numpy()
            self.values = self.batch.column(1).to_numpy(zero_copy_only=False)
            self.shifts = self.batch.column(2).to_numpy()

    def read_grid(self, grid):
        values = np.full(len(grid), np.nan)
        shifts = np.zeros(len(grid), dtype=np.int16)
        end = grid[-1] + 5000
        while self.batch is not None:
            if self.ticks[0] >= end:
                break
            mask = (self.ticks >= grid[0]) & (self.ticks < end)
            positions = ((self.ticks[mask] - grid[0]) // 5000).astype(np.intp)
            values[positions] = self.values[mask]
            shifts[positions] = self.shifts[mask]
            if self.ticks[-1] >= end:
                break
            self._advance()
        return values, shifts

    def close(self):
        self.source.close()
