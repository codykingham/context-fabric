"""
String pool management for string-valued features.

This module provides efficient storage for string-valued features using
integer indices into a shared string pool. This approach minimizes memory
usage when many nodes share the same string values.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Sentinel for missing string index
MISSING_STR_INDEX = 0xFFFFFFFF
NODE_DTYPE = 'uint32'


class StringPool:
    """
    Efficient string storage with integer indices.

    Uses numpy object arrays which support copy-on-write sharing.

    Attributes
    ----------
    strings : np.ndarray
        Array of unique strings (dtype=object)
    indices : np.ndarray
        Per-node index into strings array (dtype=uint32)
        MISSING_STR_INDEX indicates no value
    """

    def __init__(
        self,
        strings: NDArray[np.object_],
        indices: NDArray[np.uint32],
    ) -> None:
        """
        Initialize a StringPool.

        Parameters
        ----------
        strings : np.ndarray
            Array of unique strings (dtype=object)
        indices : np.ndarray
            Per-node index into strings array (dtype=uint32)
        """
        self.strings = strings
        self.indices = indices

    def get(self, node: int) -> str | None:
        """
        Get string value for node (1-indexed).

        Parameters
        ----------
        node : int
            Node number (1-indexed)

        Returns
        -------
        str | None
            String value or None if missing
        """
        # Bounds check: return None for out-of-range nodes
        arr_idx = node - 1
        if arr_idx < 0 or arr_idx >= len(self.indices):
            return None
        idx = self.indices[arr_idx]
        if idx == MISSING_STR_INDEX:
            return None
        return self.strings[idx]

    def __getitem__(self, node: int) -> str | None:
        """
        Get string value for node using bracket notation.

        Parameters
        ----------
        node : int
            Node number (1-indexed)

        Returns
        -------
        str | None
            String value or None if missing
        """
        return self.get(node)

    def __len__(self) -> int:
        """
        Number of nodes tracked.

        Returns
        -------
        int
            Length of indices array
        """
        return len(self.indices)

    def items(self) -> Iterator[tuple[int, str]]:
        """
        Iterate over (node, value) pairs efficiently using numpy.

        Only yields nodes that have values (skips MISSING entries).

        Yields
        ------
        tuple[int, str]
            (node, string_value) pairs
        """
        # Use numpy to find all nodes with values (vectorized, fast)
        mask = self.indices != MISSING_STR_INDEX
        valid_indices = np.where(mask)[0]

        for idx in valid_indices:
            node = idx + 1  # Convert 0-indexed to 1-indexed
            string_idx = self.indices[idx]
            yield (node, self.strings[string_idx])

    def to_dict(self) -> dict[int, str]:
        """
        Convert to dict efficiently.

        Returns
        -------
        dict[int, str]
            Mapping from node to string value
        """
        return dict(self.items())

    @classmethod
    def from_dict(cls, data: dict[int, str], max_node: int) -> StringPool:
        """
        Build string pool from node->string dict.

        Parameters
        ----------
        data : dict[int, str]
            Mapping from node (int) to string value
        max_node : int
            Maximum node number in corpus

        Returns
        -------
        StringPool
            New StringPool instance
        """
        # Build unique string list
        unique_strings = sorted(set(data.values()))
        string_to_idx = {s: i for i, s in enumerate(unique_strings)}

        # Build index array
        indices = np.full(max_node, MISSING_STR_INDEX, dtype=NODE_DTYPE)
        for node, value in data.items():
            indices[node - 1] = string_to_idx[value]

        strings = np.array(unique_strings, dtype=object)
        return cls(strings, indices)

    def save(self, path_prefix: str) -> None:
        """
        Save to {path_prefix}_strings.npy and {path_prefix}_idx.npy.

        Parameters
        ----------
        path_prefix : str
            Path prefix for output files
        """
        np.save(f"{path_prefix}_strings.npy", self.strings, allow_pickle=True)
        np.save(f"{path_prefix}_idx.npy", self.indices)

    @classmethod
    def load(cls, path_prefix: str, mmap_mode: str = 'r') -> StringPool:
        """
        Load from files.

        Parameters
        ----------
        path_prefix : str
            Path prefix for input files
        mmap_mode : str, optional
            Memory-map mode for indices array (default: 'r')

        Returns
        -------
        StringPool
            Loaded StringPool instance
        """
        # Note: object arrays can't be mmap'd, but they're typically small
        strings = np.load(f"{path_prefix}_strings.npy", allow_pickle=True)
        indices = np.load(f"{path_prefix}_idx.npy", mmap_mode=mmap_mode)
        return cls(strings, indices)

    def get_value_index(self, value: str) -> int | None:
        """
        Get the internal index for a string value.

        Useful for pre-computing indices for repeated filtering operations.

        Parameters
        ----------
        value : str
            String value to look up

        Returns
        -------
        int | None
            Internal index, or None if value doesn't exist
        """
        # Linear search through strings array (typically small)
        for i, s in enumerate(self.strings):
            if s == value:
                return i
        return None

    def filter_by_value(
        self, nodes: list[int] | range, value: str
    ) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes where feature equals value.

        This is much faster than calling get() in a loop because it uses
        numpy vectorized operations instead of per-element Python calls.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)
        value : str
            Value to match

        Returns
        -------
        NDArray[np.int64]
            Array of matching nodes (1-indexed)
        """
        if not nodes:
            return np.array([], dtype=np.int64)

        # Find the index for this value
        value_idx = self.get_value_index(value)
        if value_idx is None:
            return np.array([], dtype=np.int64)

        # Convert nodes to 0-indexed array indices
        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        # Bounds check: only keep valid indices
        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.indices))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        # Vectorized lookup and comparison
        values_at_nodes = self.indices[valid_arr_indices]
        match_mask = values_at_nodes == value_idx

        return valid_nodes[match_mask]

    def filter_by_values(
        self, nodes: list[int] | range, values: set[str]
    ) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes where feature is in values set.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)
        values : set[str]
            Set of values to match

        Returns
        -------
        NDArray[np.int64]
            Array of matching nodes (1-indexed)
        """
        if not nodes or not values:
            return np.array([], dtype=np.int64)

        # Find indices for all values
        value_indices = set()
        for v in values:
            idx = self.get_value_index(v)
            if idx is not None:
                value_indices.add(idx)

        if not value_indices:
            return np.array([], dtype=np.int64)

        # Convert nodes to 0-indexed array indices
        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        # Bounds check
        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.indices))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        # Vectorized lookup
        values_at_nodes = self.indices[valid_arr_indices]

        # Check if each value is in the target set
        match_mask = np.isin(values_at_nodes, list(value_indices))

        return valid_nodes[match_mask]

    def filter_has_value(self, nodes: list[int] | range) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes that have any value.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)

        Returns
        -------
        NDArray[np.int64]
            Array of nodes with values (1-indexed)
        """
        if not nodes:
            return np.array([], dtype=np.int64)

        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.indices))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        values_at_nodes = self.indices[valid_arr_indices]
        has_value_mask = values_at_nodes != MISSING_STR_INDEX

        return valid_nodes[has_value_mask]

    def filter_missing_value(self, nodes: list[int] | range) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes that have no value.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)

        Returns
        -------
        NDArray[np.int64]
            Array of nodes without values (1-indexed)
        """
        if not nodes:
            return np.array([], dtype=np.int64)

        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.indices))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        values_at_nodes = self.indices[valid_arr_indices]
        missing_mask = values_at_nodes == MISSING_STR_INDEX

        return valid_nodes[missing_mask]

    def get_frequency_counts(self) -> dict[str, int]:
        """
        Get frequency counts of all values using vectorized numpy operations.

        Returns
        -------
        dict[str, int]
            Mapping from string value to count
        """
        valid_mask = self.indices != MISSING_STR_INDEX
        valid_indices = self.indices[valid_mask]
        unique_idx, counts = np.unique(valid_indices, return_counts=True)
        return {self.strings[idx]: int(count) for idx, count in zip(unique_idx, counts)}


class IntFeatureArray:
    """
    Integer feature storage.

    Dense array with sentinel for missing values.

    Attributes
    ----------
    values : np.ndarray
        Array of integer values (dtype=int32)
        MISSING (-1) indicates no value
    """

    MISSING = -1

    def __init__(self, values: NDArray[np.int32]) -> None:
        """
        Initialize an IntFeatureArray.

        Parameters
        ----------
        values : np.ndarray
            Array of integer values (dtype=int32)
        """
        self.values = values

    def get(self, node: int) -> int | None:
        """
        Get int value for node (1-indexed).

        Parameters
        ----------
        node : int
            Node number (1-indexed)

        Returns
        -------
        int | None
            Integer value or None if missing
        """
        # Bounds check: return None for out-of-range nodes
        arr_idx = node - 1
        if arr_idx < 0 or arr_idx >= len(self.values):
            return None
        val = self.values[arr_idx]
        if val == self.MISSING:
            return None
        return int(val)

    def __getitem__(self, node: int) -> int | None:
        """
        Get int value for node using bracket notation.

        Parameters
        ----------
        node : int
            Node number (1-indexed)

        Returns
        -------
        int | None
            Integer value or None if missing
        """
        return self.get(node)

    def __len__(self) -> int:
        """
        Number of nodes tracked.

        Returns
        -------
        int
            Length of values array
        """
        return len(self.values)

    def items(self) -> Iterator[tuple[int, int]]:
        """
        Iterate over (node, value) pairs efficiently using numpy.

        Only yields nodes that have values (skips MISSING entries).

        Yields
        ------
        tuple[int, int]
            (node, int_value) pairs
        """
        # Use numpy to find all nodes with values (vectorized, fast)
        mask = self.values != self.MISSING
        valid_indices = np.where(mask)[0]

        for idx in valid_indices:
            node = idx + 1  # Convert 0-indexed to 1-indexed
            yield (node, int(self.values[idx]))

    def to_dict(self) -> dict[int, int]:
        """
        Convert to dict efficiently.

        Returns
        -------
        dict[int, int]
            Mapping from node to int value
        """
        return dict(self.items())

    @classmethod
    def from_dict(cls, data: dict[int, int | None], max_node: int) -> IntFeatureArray:
        """
        Build from node->int dict.

        Parameters
        ----------
        data : dict[int, int | None]
            Mapping from node (int) to integer value (or None for missing)
        max_node : int
            Maximum node number in corpus

        Returns
        -------
        IntFeatureArray
            New IntFeatureArray instance
        """
        values = np.full(max_node, cls.MISSING, dtype='int32')
        for node, value in data.items():
            # None values stay as MISSING sentinel
            if value is not None:
                values[node - 1] = value
        return cls(values)

    def save(self, path: str) -> None:
        """
        Save to .npy file.

        Parameters
        ----------
        path : str
            Output file path
        """
        np.save(path, self.values)

    @classmethod
    def load(cls, path: str, mmap_mode: str = 'r') -> IntFeatureArray:
        """
        Load from .npy file.

        Parameters
        ----------
        path : str
            Input file path
        mmap_mode : str, optional
            Memory-map mode (default: 'r')

        Returns
        -------
        IntFeatureArray
            Loaded IntFeatureArray instance
        """
        values = np.load(path, mmap_mode=mmap_mode)
        return cls(values)

    def filter_by_value(
        self, nodes: list[int] | range, value: int
    ) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes where feature equals value.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)
        value : int
            Value to match

        Returns
        -------
        NDArray[np.int64]
            Array of matching nodes (1-indexed)
        """
        if not nodes:
            return np.array([], dtype=np.int64)

        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.values))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        values_at_nodes = self.values[valid_arr_indices]
        match_mask = values_at_nodes == value

        return valid_nodes[match_mask]

    def filter_by_values(
        self, nodes: list[int] | range, values: set[int]
    ) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes where feature is in values set.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)
        values : set[int]
            Set of values to match

        Returns
        -------
        NDArray[np.int64]
            Array of matching nodes (1-indexed)
        """
        if not nodes or not values:
            return np.array([], dtype=np.int64)

        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.values))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        values_at_nodes = self.values[valid_arr_indices]
        match_mask = np.isin(values_at_nodes, list(values))

        return valid_nodes[match_mask]

    def filter_less_than(
        self, nodes: list[int] | range, threshold: int
    ) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes where value < threshold.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)
        threshold : int
            Threshold value

        Returns
        -------
        NDArray[np.int64]
            Array of matching nodes (1-indexed)
        """
        if not nodes:
            return np.array([], dtype=np.int64)

        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.values))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        values_at_nodes = self.values[valid_arr_indices]
        # Must have a value AND be less than threshold
        match_mask = (values_at_nodes != self.MISSING) & (values_at_nodes < threshold)

        return valid_nodes[match_mask]

    def filter_greater_than(
        self, nodes: list[int] | range, threshold: int
    ) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes where value > threshold.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)
        threshold : int
            Threshold value

        Returns
        -------
        NDArray[np.int64]
            Array of matching nodes (1-indexed)
        """
        if not nodes:
            return np.array([], dtype=np.int64)

        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.values))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        values_at_nodes = self.values[valid_arr_indices]
        # Must have a value AND be greater than threshold
        match_mask = (values_at_nodes != self.MISSING) & (values_at_nodes > threshold)

        return valid_nodes[match_mask]

    def filter_has_value(self, nodes: list[int] | range) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes that have any value.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)

        Returns
        -------
        NDArray[np.int64]
            Array of nodes with values (1-indexed)
        """
        if not nodes:
            return np.array([], dtype=np.int64)

        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.values))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        values_at_nodes = self.values[valid_arr_indices]
        has_value_mask = values_at_nodes != self.MISSING

        return valid_nodes[has_value_mask]

    def filter_missing_value(self, nodes: list[int] | range) -> NDArray[np.int64]:
        """
        Vectorized filter: return nodes that have no value.

        Parameters
        ----------
        nodes : list[int] | range
            Nodes to filter (1-indexed)

        Returns
        -------
        NDArray[np.int64]
            Array of nodes without values (1-indexed)
        """
        if not nodes:
            return np.array([], dtype=np.int64)

        node_arr = np.asarray(nodes, dtype=np.int64)
        arr_indices = node_arr - 1

        valid_mask = (arr_indices >= 0) & (arr_indices < len(self.values))
        valid_arr_indices = arr_indices[valid_mask]
        valid_nodes = node_arr[valid_mask]

        values_at_nodes = self.values[valid_arr_indices]
        missing_mask = values_at_nodes == self.MISSING

        return valid_nodes[missing_mask]

    def get_frequency_counts(self) -> dict[int, int]:
        """
        Get frequency counts of all values using vectorized numpy operations.

        Returns
        -------
        dict[int, int]
            Mapping from integer value to count
        """
        valid_mask = self.values != self.MISSING
        valid_values = self.values[valid_mask]
        unique_vals, counts = np.unique(valid_values, return_counts=True)
        return {int(val): int(count) for val, count in zip(unique_vals, counts)}
