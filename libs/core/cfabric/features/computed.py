"""
# Pre-computed data components

In order to make the API work, TF prepares some data and saves it in
quick-load format. Most of this data are the features, but there is some extra
data needed for the special functions of the `tf.parameters.WARP` features and the
`cfabric.locality.Locality` API.

Normally, you do not use this data, but since it is there, it might be valuable,
so we have made it accessible in the `cfabric.computed.Computeds`-API.

!!! explanation "Pre-computed data storage"
    Pre-computed data is stored in a `.cfm` directory (Context Fabric Mmap format)
    inside the directory where the `otype` feature is encountered.

    The `.cfm` format uses memory-mapped numpy arrays for:
    - Shared memory across async workers
    - Reduced memory footprint
    - Near-zero startup time after initial compilation

    Use `TF.compile()` to generate the `.cfm` directory from `.tf` source files.
    Subsequent calls to `TF.load()` will automatically use the compiled format.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cfabric.core.api import Api
    from cfabric.storage.csr import CSRArray


class Computeds:
    pass


class Computed:
    """Provides access to pre-computed data.

    For component `ccc` it is the result of `C.ccc` or `Cs('ccc')`.
    """

    def __init__(self, api: Api, data: Any) -> None:
        self.api = api
        self.data = data


class RankComputed(Computed):
    """C.rank: canonical position of each node.

    Supports numpy array backend for mmap.
    data[n-1] gives rank of node n.
    """

    def __getitem__(self, n: int) -> int:
        idx = n - 1
        if idx < 0 or idx >= len(self.data):
            # Out-of-bounds nodes get a rank beyond the last valid node
            return len(self.data) + 1
        return int(self.data[idx])


class OrderComputed(Computed):
    """C.order: nodes in canonical order.

    Supports numpy array backend for mmap.
    data[i] gives node at position i.
    """

    def __getitem__(self, i: int) -> int:
        return int(self.data[i])

    def __iter__(self) -> Iterator[int]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


class LevUpComputed(Computed):
    """C.levUp: embedders of each node.

    Supports CSRArray backend for mmap.
    """

    def __getitem__(self, n: int) -> tuple[int, ...]:
        from cfabric.storage.csr import CSRArray

        if isinstance(self.data, CSRArray):
            return self.data.get_as_tuple(n - 1)
        return self.data[n - 1]


class LevDownComputed(Computed):
    """C.levDown: embeddees of each node.

    Supports CSRArray backend for mmap.
    Only for non-slot nodes, so index is n - maxSlot - 1.
    """

    def __getitem__(self, n: int) -> tuple[int, ...]:
        from cfabric.storage.csr import CSRArray

        maxSlot = self.api.F.otype.maxSlot
        idx = n - maxSlot - 1
        if idx < 0:
            return ()  # Slots have no embeddees

        if isinstance(self.data, CSRArray):
            return self.data.get_as_tuple(idx)
        return self.data[idx]


class LevelsComputed(Computed):
    """C.levels: node type hierarchy data."""

    pass


class BoundaryComputed(Computed):
    """C.boundary: first/last slot boundary data."""

    pass
