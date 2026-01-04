"""
Compressed Sparse Row (CSR) utilities for variable-length data.
"""

import numpy as np
from typing import Tuple, Sequence

# Node indexing dtypes
NODE_DTYPE = 'uint32'
INDEX_DTYPE = 'uint32'


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

    def __init__(self, indptr: np.ndarray, data: np.ndarray):
        self.indptr = indptr
        self.data = data

    def __getitem__(self, i: int) -> tuple:
        """Get data for row i as tuple."""
        return tuple(self.data[self.indptr[i]:self.indptr[i + 1]])

    def get_as_tuple(self, i: int) -> tuple:
        """Get data for row i as tuple (alias for __getitem__)."""
        return self[i]

    def __len__(self) -> int:
        return len(self.indptr) - 1

    @classmethod
    def from_sequences(cls, sequences: Sequence[Sequence[int]]) -> 'CSRArray':
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

    def save(self, path_prefix: str):
        """Save to {path_prefix}_indptr.npy and {path_prefix}_data.npy"""
        np.save(f"{path_prefix}_indptr.npy", self.indptr)
        np.save(f"{path_prefix}_data.npy", self.data)

    @classmethod
    def load(cls, path_prefix: str, mmap_mode: str = 'r') -> 'CSRArray':
        """Load from files."""
        indptr = np.load(f"{path_prefix}_indptr.npy", mmap_mode=mmap_mode)
        data = np.load(f"{path_prefix}_data.npy", mmap_mode=mmap_mode)
        return cls(indptr, data)


class CSRArrayWithValues(CSRArray):
    """CSR with associated values (for edge features with values)."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, values: np.ndarray):
        super().__init__(indptr, indices)
        self.indices = indices  # alias for clarity
        self.values = values

    def __getitem__(self, i: int) -> Tuple[tuple, tuple]:
        """Get (indices, values) for row i as tuples."""
        start, end = self.indptr[i], self.indptr[i + 1]
        return tuple(self.indices[start:end]), tuple(self.values[start:end])

    def get_as_dict(self, i: int) -> dict:
        """Get as {index: value} dict for row i."""
        indices, values = self[i]
        return dict(zip(indices, values))

    def save(self, path_prefix: str):
        """Save to files including values."""
        np.save(f"{path_prefix}_indptr.npy", self.indptr)
        np.save(f"{path_prefix}_indices.npy", self.indices)
        np.save(f"{path_prefix}_values.npy", self.values)

    @classmethod
    def load(cls, path_prefix: str, mmap_mode: str = 'r') -> 'CSRArrayWithValues':
        """Load from files."""
        indptr = np.load(f"{path_prefix}_indptr.npy", mmap_mode=mmap_mode)
        indices = np.load(f"{path_prefix}_indices.npy", mmap_mode=mmap_mode)
        values = np.load(f"{path_prefix}_values.npy", mmap_mode=mmap_mode)
        return cls(indptr, indices, values)

    @classmethod
    def from_dict_of_dicts(cls, data: dict, num_rows: int, value_dtype='int32') -> 'CSRArrayWithValues':
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
