"""
Memory-mapped array management for Context Fabric.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

from .csr import CSRArray
from .strings import StringPool


class MmapManager:
    """
    Manages memory-mapped numpy arrays for a corpus.

    Provides lazy loading and shared access to corpus data.

    Parameters
    ----------
    cfm_path : Path
        Path to .cfm/{version}/ directory
    """

    def __init__(self, cfm_path: Path):
        """
        Initialize manager for a .cfm directory.

        Parameters
        ----------
        cfm_path : Path
            Path to .cfm/{version}/ directory
        """
        self.cfm_path = Path(cfm_path)
        self._arrays: Dict[str, np.ndarray] = {}
        self._meta: Optional[dict] = None

    @property
    def meta(self) -> dict:
        """Load and cache corpus metadata."""
        if self._meta is None:
            with open(self.cfm_path / 'meta.json') as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def max_slot(self) -> int:
        return self.meta['max_slot']

    @property
    def max_node(self) -> int:
        return self.meta['max_node']

    @property
    def slot_type(self) -> str:
        return self.meta['slot_type']

    @property
    def node_types(self) -> list:
        return self.meta['node_types']

    def get_array(self, *path_parts: str) -> np.ndarray:
        """
        Get a memory-mapped array, loading lazily.

        Parameters
        ----------
        path_parts : str
            Path components relative to cfm_path
            e.g., get_array('warp', 'otype') → warp/otype.npy

        Returns
        -------
        np.ndarray
            Memory-mapped array (read-only)
        """
        key = '/'.join(path_parts)
        if key not in self._arrays:
            file_path = self.cfm_path.joinpath(*path_parts[:-1]) / f"{path_parts[-1]}.npy"
            self._arrays[key] = np.load(file_path, mmap_mode='r')
        return self._arrays[key]

    def get_json(self, *path_parts: str) -> Any:
        """Load a JSON metadata file."""
        file_path = self.cfm_path.joinpath(*path_parts[:-1]) / f"{path_parts[-1]}.json"
        with open(file_path) as f:
            return json.load(f)

    def get_string_pool(self, feature_name: str) -> StringPool:
        """Get string pool for a string-valued feature."""
        return StringPool.load(
            str(self.cfm_path / 'features' / feature_name),
            mmap_mode='r'
        )

    def get_csr(self, *path_parts: str) -> CSRArray:
        """Get CSR array pair."""
        base_path = self.cfm_path.joinpath(*path_parts[:-1]) / path_parts[-1]
        return CSRArray.load(str(base_path), mmap_mode='r')

    def exists(self) -> bool:
        """Check if the .cfm directory exists and has metadata."""
        return (self.cfm_path / 'meta.json').exists()

    def close(self):
        """Release all memory mappings."""
        self._arrays.clear()
        self._meta = None
