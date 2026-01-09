"""
Compressed Sparse Row (CSR) utilities for variable-length data.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# Node indexing dtypes
NODE_DTYPE = 'uint32'
INDEX_DTYPE = 'uint32'

# Environment variable to control embedding cache behavior
# Values: "on" (default), "off"
# Set CF_EMBEDDING_CACHE=off to disable automatic preloading
_EMBEDDING_CACHE_MODE = os.environ.get('CF_EMBEDDING_CACHE', 'on').lower()


class CSRArray:
    """
    CSR representation for variable-length node data.

    For data where each node maps to a variable number of values
    (e.g., oslots, edges, levUp/levDown).

    Attributes
    ----------
    indptr : np.ndarray
        Index pointers. Row i contains data[indptr[i]:indptr[i+1]]
    data : np.ndarray
        Concatenated row data
    """

    def __init__(self, indptr: NDArray[np.uint32], data: NDArray[np.uint32]) -> None:
        self._indptr = indptr
        self._data = data
        self._ram_indptr: NDArray[np.uint32] | None = None
        self._ram_data: NDArray[np.uint32] | None = None

    @property
    def indptr(self) -> NDArray[np.uint32]:
        """Return indptr, using RAM cache if available."""
        return self._ram_indptr if self._ram_indptr is not None else self._indptr

    @property
    def data(self) -> NDArray[np.uint32]:
        """Return data, using RAM cache if available."""
        return self._ram_data if self._ram_data is not None else self._data

    @property
    def is_cached(self) -> bool:
        """Return True if data is cached in RAM."""
        return self._ram_indptr is not None

    def preload_to_ram(self) -> None:
        """Load CSR data into RAM for faster access.

        This trades memory for speed. Use when you need fast repeated
        access to the CSR data (e.g., embedding queries).

        Memory cost: indptr.nbytes + data.nbytes (typically 50-100MB for BHSA)
        """
        if self._ram_indptr is None:
            self._ram_indptr = np.array(self._indptr)
            self._ram_data = np.array(self._data)

    def release_cache(self) -> None:
        """Release RAM cache, returning to mmap-only access."""
        self._ram_indptr = None
        self._ram_data = None

    def memory_usage_bytes(self) -> int:
        """Return memory used by RAM cache, or 0 if not cached."""
        if self._ram_indptr is None:
            return 0
        return self._ram_indptr.nbytes + self._ram_data.nbytes

    def __getitem__(self, i: int) -> tuple[int, ...]:
        """Get data for row i as tuple."""
        return tuple(self.data[self.indptr[i]:self.indptr[i + 1]])

    def get_as_tuple(self, i: int) -> tuple[int, ...]:
        """Get data for row i as tuple (alias for __getitem__)."""
        return self[i]

    def __len__(self) -> int:
        return len(self.indptr) - 1

    @classmethod
    def from_sequences(cls, sequences: Sequence[Sequence[int]]) -> CSRArray:
        """
        Build CSR from sequence of sequences.

        Parameters
        ----------
        sequences : Sequence[Sequence[int]]
            List of variable-length integer sequences

        Returns
        -------
        CSRArray
        """
        indptr = np.zeros(len(sequences) + 1, dtype=INDEX_DTYPE)
        total = sum(len(s) for s in sequences)
        data = np.zeros(total, dtype=NODE_DTYPE)

        offset = 0
        for i, seq in enumerate(sequences):
            indptr[i] = offset
            for j, val in enumerate(seq):
                data[offset + j] = val
            offset += len(seq)
        indptr[-1] = offset

        return cls(indptr, data)

    def save(self, path_prefix: str) -> None:
        """Save to {path_prefix}_indptr.npy and {path_prefix}_data.npy"""
        np.save(f"{path_prefix}_indptr.npy", self.indptr)
        np.save(f"{path_prefix}_data.npy", self.data)

    @classmethod
    def load(cls, path_prefix: str, mmap_mode: str = 'r') -> CSRArray:
        """Load from files."""
        indptr = np.load(f"{path_prefix}_indptr.npy", mmap_mode=mmap_mode)
        data = np.load(f"{path_prefix}_data.npy", mmap_mode=mmap_mode)
        return cls(indptr, data)

    def get_all_targets(self, sources: set[int]) -> set[int]:
        """Get union of all targets for a set of source nodes.

        Parameters
        ----------
        sources : set[int]
            Source node IDs (1-indexed)

        Returns
        -------
        set[int]
            Union of all targets from all sources
        """
        if not sources:
            return set()

        # Convert to 0-indexed array indices
        source_arr = np.array(list(sources), dtype=np.int64) - 1
        valid_mask = (source_arr >= 0) & (source_arr < len(self))
        valid_sources = source_arr[valid_mask]

        if len(valid_sources) == 0:
            return set()

        # Collect all targets using vectorized slicing
        all_targets = []
        for idx in valid_sources:
            start, end = self.indptr[idx], self.indptr[idx + 1]
            if start < end:
                all_targets.append(self.data[start:end])

        if not all_targets:
            return set()

        return set(np.concatenate(all_targets).tolist())

    def filter_sources_with_targets_in(
        self, sources: set[int], target_set: set[int]
    ) -> tuple[set[int], set[int]]:
        """Filter sources that have at least one target in target_set.

        Returns both the filtered sources AND the matched targets.
        This is the core operation for relation spinning.

        Parameters
        ----------
        sources : set[int]
            Source node IDs (1-indexed)
        target_set : set[int]
            Target node IDs to match against (1-indexed)

        Returns
        -------
        tuple[set[int], set[int]]
            (filtered_sources, matched_targets) - sources with matches and targets that were matched
        """
        if not sources or not target_set:
            return set(), set()

        matched_sources = set()
        matched_targets = set()
        csr_len = len(self)

        # Direct iteration with cached indptr/data references
        indptr = self.indptr
        data = self.data

        for src in sources:
            idx = src - 1
            if idx < 0 or idx >= csr_len:
                continue

            start = indptr[idx]
            end = indptr[idx + 1]
            if start >= end:
                continue

            # Get targets for this source
            targets = data[start:end]

            # Check which targets are in the target set
            for t in targets:
                if t in target_set:
                    matched_sources.add(src)
                    matched_targets.add(int(t))

        return matched_sources, matched_targets

class CSRArrayWithValues(CSRArray):
    """CSR with associated values (for edge features with values)."""

    def __init__(
        self,
        indptr: NDArray[np.uint32],
        indices: NDArray[np.uint32],
        values: NDArray[Any],
    ) -> None:
        super().__init__(indptr, indices)
        self.indices = indices  # alias for clarity
        self.values = values

    def __getitem__(self, i: int) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        """Get (indices, values) for row i as tuples."""
        start, end = self.indptr[i], self.indptr[i + 1]
        return tuple(self.indices[start:end]), tuple(self.values[start:end])

    def get_as_dict(self, i: int) -> dict[int, Any]:
        """Get as {index: value} dict for row i."""
        indices, values = self[i]
        return dict(zip(indices, values))

    def save(self, path_prefix: str) -> None:
        """Save to files including values (with string encoding if needed)."""
        import json
        from pathlib import Path

        np.save(f"{path_prefix}_indptr.npy", self.indptr)
        np.save(f"{path_prefix}_indices.npy", self.indices)

        if self.values.dtype == np.object_:
            # Encode strings as integer indices (object arrays can't be mmap'd)
            unique_values = list(dict.fromkeys(self.values))  # preserve order, dedupe
            value_to_idx = {v: i for i, v in enumerate(unique_values)}
            encoded = np.array([value_to_idx[v] for v in self.values], dtype=np.uint32)
            np.save(f"{path_prefix}_values.npy", encoded)

            with open(f"{path_prefix}_values_lookup.json", 'w') as f:
                json.dump(unique_values, f)
        else:
            np.save(f"{path_prefix}_values.npy", self.values)

    @classmethod
    def load(cls, path_prefix: str, mmap_mode: str = 'r') -> CSRArrayWithValues:
        """Load from files (with string decoding if needed)."""
        import json
        from pathlib import Path

        indptr = np.load(f"{path_prefix}_indptr.npy", mmap_mode=mmap_mode)
        indices = np.load(f"{path_prefix}_indices.npy", mmap_mode=mmap_mode)

        lookup_path = Path(f"{path_prefix}_values_lookup.json")
        if lookup_path.exists():
            # String values: load lookup and decode
            encoded = np.load(f"{path_prefix}_values.npy", mmap_mode=mmap_mode)
            with open(lookup_path) as f:
                lookup = json.load(f)
            values = np.array([lookup[i] for i in encoded], dtype=object)
        else:
            values = np.load(f"{path_prefix}_values.npy", mmap_mode=mmap_mode)

        return cls(indptr, indices, values)

    @classmethod
    def from_dict_of_dicts(
        cls,
        data: dict[int, dict[int, Any]],
        num_rows: int,
        value_dtype: str = 'int32',
    ) -> CSRArrayWithValues:
        """
        Build from dict[int, dict[int, value]].

        Parameters
        ----------
        data : dict
            Mapping from row index to {column: value} dict
        num_rows : int
            Total number of rows
        value_dtype : str
            Numpy dtype for values

        Returns
        -------
        CSRArrayWithValues
        """
        # Count total entries
        total = sum(len(d) for d in data.values())

        indptr = np.zeros(num_rows + 1, dtype=INDEX_DTYPE)
        indices = np.zeros(total, dtype=NODE_DTYPE)
        values = np.zeros(total, dtype=value_dtype)

        offset = 0
        for i in range(num_rows):
            indptr[i] = offset
            if i in data:
                row_data = data[i]
                for j, (col, val) in enumerate(sorted(row_data.items())):
                    indices[offset + j] = col
                    values[offset + j] = val
                offset += len(row_data)
        indptr[-1] = offset

        return cls(indptr, indices, values)
