"""
# Mappings from nodes to values.

Every node feature is logically a mapping from nodes to values,
string or integer.

A feature object gives you methods that you can pass a node and that returns
its value for that node.

It is easiest to think of all node features as a dictionary keyed by nodes.

However, some features have an optimised representation, and do not have
a dictionary underneath.

But you can still iterate over the data of a feature as if it were a
dictionary: `cfabric.nodefeature.NodeFeature.items`
"""

from __future__ import annotations

import collections
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from cfabric.storage.string_pool import StringPool, IntFeatureArray
from cfabric.utils.helpers import safe_rank_key

if TYPE_CHECKING:
    from cfabric.core.api import Api


class NodeFeatures:
    pass


class NodeFeature:
    """Provides access to (node) feature data.

    For feature `fff` it is the result of `F.fff` or `Fs('fff')`.

    Supports both dict-based storage (.tf loading) and mmap-based backends (.cfm loading)
    (StringPool for string features, IntFeatureArray for int features).
    """

    def __init__(
        self,
        api: Api,
        metaData: dict[str, str],
        data: dict[int, str | int] | StringPool | IntFeatureArray,
    ) -> None:
        self.api = api
        self.meta = metaData
        """Metadata of the feature.

        This is the information found in the lines starting with `@`
        in the `.tf` feature file.
        """

        self._data = data
        self._is_mmap = isinstance(data, (StringPool, IntFeatureArray))
        self._cached_data: dict[int, str | int] | None = None  # Cache for materialized dict

    @property
    def data(self) -> dict[int, str | int]:
        """Get data as dict (for backward compatibility).

        Note: For mmap backends, this materializes the data into memory.
        Use v() for efficient single lookups.

        Returns
        -------
        dict
            The feature data as a dictionary mapping nodes to values.
        """
        if self._is_mmap:
            return self._materialize()
        return self._data

    def _materialize(self) -> dict[int, str | int]:
        """Convert mmap storage to dict (cached).

        Returns
        -------
        dict
            Dictionary mapping nodes to their values.
        """
        if self._cached_data is not None:
            return self._cached_data

        # Use efficient items() from StringPool/IntFeatureArray
        # which uses numpy vectorized operations to find non-missing values
        self._cached_data = dict(self._data.items())
        return self._cached_data

    def items(self) -> Iterator[tuple[int, str | int]]:
        """A generator that yields the items of the feature, seen as a mapping.

        It does not yield entries for nodes without values,
        so this gives you a rather efficient way to iterate over
        just the feature data, instead of over all nodes.

        If you need this repeatedly, or you need the whole dictionary,
        you can store the result as follows:

           data = dict(F.fff.items())

        """
        # Both dict and mmap backends (StringPool/IntFeatureArray) have items()
        return self._data.items()

    def v(self, n: int) -> str | int | None:
        """Get the value of a feature for a node.

        Parameters
        ----------
        n: integer
            The node in question

        Returns
        -------
        integer | string | None
            The value of the feature for that node, if it is defined, else `None`.
        """
        if self._is_mmap:
            return self._data.get(n)

        if n in self._data:
            return self._data[n]
        return None

    def s(self, val: str | int) -> tuple[int, ...]:
        """Query all nodes having a specified feature value.

        This is an other way to walk through nodes than using
        `cfabric.nodes.Nodes.walk`.

        Parameters
        ----------
        value: integer | string
            The feature value that all resulting nodes have.

        Returns
        -------
        tuple of integer
            All nodes that have this value for this feature,
            sorted in the canonical order.
            (`cfabric.nodes`)
        """

        rank_key = safe_rank_key(self.api.C.rank.data)

        if self._is_mmap:
            # Use vectorized filtering for mmap-backed features
            max_node = len(self._data)
            all_nodes = range(1, max_node + 1)
            matches = self._data.filter_by_value(all_nodes, val)
            return tuple(sorted(matches, key=rank_key))
        else:
            return tuple(
                sorted(
                    [n for n in self._data if self._data[n] == val],
                    key=rank_key,
                )
            )

    def freqList(self, nodeTypes: set[str] | None = None) -> tuple[tuple[str | int, int], ...]:
        """Frequency list of the values of this feature.

        Inspect the values of this feature and see how often they occur.

        Parameters
        ----------
        nodeTypes: set of string, optional None
            If you pass a set of node types, only the values for nodes
            within those types will be counted.

        Returns
        -------
        tuple of 2-tuple
            A tuple of `(value, frequency)`, items, ordered by `frequency`,
            highest frequencies first.

        """
        fql: collections.Counter[str | int] = collections.Counter()

        if self._is_mmap and nodeTypes is None:
            # Fast path: vectorized frequency counting
            fql.update(self._data.get_frequency_counts())
        elif self._is_mmap:
            # Mmap with type filtering - need per-node iteration
            fOtype = self.api.F.otype.v
            max_node = len(self._data)
            for n in range(1, max_node + 1):
                val = self.v(n)
                if val is not None and fOtype(n) in nodeTypes:
                    fql[val] += 1
        elif nodeTypes is None:
            # Dict-backed, no filtering
            for n in self._data:
                fql[self._data[n]] += 1
        else:
            # Dict-backed with type filtering
            fOtype = self.api.F.otype.v
            for n in self._data:
                if fOtype(n) in nodeTypes:
                    fql[self._data[n]] += 1

        return tuple(sorted(fql.items(), key=lambda x: (-x[1], x[0])))
