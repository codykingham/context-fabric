"""
Mappings from edges to values.

Every edge feature is logically a mapping from pairs of nodes to values,
string or integer.

A feature object gives you methods that you can pass a node and that returns
its value for that node.

It is easiest to think of all edge features as a dictionary keyed by nodes.
The values are either sets or dictionaries.
If the value is a set, then the elements are the second node in the pair
and the value is `None`.
If the value is a dictionary, then the keys are the second node in the pair,
and the value is the value that the edge feature assigns to this pair.

However, some features have an optimised representation, and do not have
a dictionary underneath.

But you can still iterate over the data of a feature as if it were a
dictionary: `cfabric.edgefeature.EdgeFeature.items`

This module supports two storage backends:
- Legacy dict-based storage: dict[int, set|dict]
- CSR mmap-based storage: CSRArray or CSRArrayWithValues
"""

from __future__ import annotations

import collections
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from cfabric.utils.helpers import makeInverse, makeInverseVal
from cfabric.storage.csr import CSRArray, CSRArrayWithValues

if TYPE_CHECKING:
    from cfabric.core.api import Api


class EdgeFeatures:
    pass


class EdgeFeature:
    """Provides access to (edge) feature data.

    For feature `fff` it is the result of `E.fff` or `Es('fff')`.

    This class supports two storage backends:
    - Legacy: dict-based storage (dict[int, set|dict])
    - Mmap: CSRArray or CSRArrayWithValues for memory-mapped access

    The backend is auto-detected based on the data type passed to __init__.
    """

    def __init__(
        self,
        api: Api,
        metaData: dict[str, Any],
        data: dict[int, set[int] | dict[int, Any]] | CSRArray | CSRArrayWithValues | tuple[Any, Any],
        doValues: bool,
        dataInv: CSRArray | CSRArrayWithValues | None = None,
    ) -> None:
        self.api = api
        self.meta = metaData
        """Metadata of the feature.

        This is the information found in the lines starting with `@`
        in the `.tf` feature file.
        """

        self.doValues = doValues

        # For mmap backend with int values, get sentinel for None values
        # The sentinel is stored in metadata during compilation
        self._none_sentinel = metaData.get('none_sentinel')

        # Detect backend type
        if isinstance(data, (CSRArray, CSRArrayWithValues)):
            # CSR mmap backend
            self._is_mmap = True
            self._data = data
            self._dataInv = dataInv  # Must be provided for mmap backend
        elif isinstance(data, tuple) and len(data) == 2:
            # Legacy tuple format: (data, dataInv)
            self._is_mmap = False
            self._data = data[0]
            self._dataInv = data[1]
        else:
            # Legacy dict format
            self._is_mmap = False
            self._data = data
            self._dataInv = (
                makeInverseVal(self._data) if doValues else makeInverse(self._data)
            )

    def _convert_sentinel_to_none(self, val: Any) -> Any:
        """Convert sentinel value back to None for int edge values."""
        if self._none_sentinel is not None and val == self._none_sentinel:
            return None
        return val

    def _convert_dict_sentinels(self, d: dict[int, Any]) -> dict[int, Any]:
        """Convert all sentinel values in a dict to None."""
        if self._none_sentinel is None:
            return d
        return {k: (None if v == self._none_sentinel else v) for k, v in d.items()}

    @property
    def data(self) -> dict[int, set[int] | dict[int, Any]]:
        """Get forward edge data.

        For legacy backend, returns the dict directly.
        For mmap backend, materializes CSR to dict (for backward compatibility).
        """
        if self._is_mmap:
            return self._materialize_forward()
        return self._data

    @property
    def dataInv(self) -> dict[int, set[int] | dict[int, Any]]:
        """Get inverse edge data.

        For legacy backend, returns the dict directly.
        For mmap backend, materializes CSR to dict (for backward compatibility).
        """
        if self._is_mmap:
            return self._materialize_inverse()
        return self._dataInv

    def _materialize_forward(self) -> dict[int, set[int] | dict[int, Any]]:
        """Convert forward CSR data to dict format."""
        result = {}
        csr = self._data
        for i in range(len(csr)):
            n = i + 1  # 0-indexed CSR to 1-indexed nodes
            if isinstance(csr, CSRArrayWithValues):
                indices, values = csr[i]
                if len(indices) > 0:
                    d = dict(zip(indices.tolist(), values.tolist()))
                    result[n] = self._convert_dict_sentinels(d)
            else:
                targets = csr[i]
                if len(targets) > 0:
                    result[n] = set(targets.tolist())
        return result

    def _materialize_inverse(self) -> dict[int, set[int] | dict[int, Any]]:
        """Convert inverse CSR data to dict format."""
        result = {}
        csr = self._dataInv
        if csr is None:
            return result
        for i in range(len(csr)):
            n = i + 1  # 0-indexed CSR to 1-indexed nodes
            if isinstance(csr, CSRArrayWithValues):
                indices, values = csr[i]
                if len(indices) > 0:
                    d = dict(zip(indices.tolist(), values.tolist()))
                    result[n] = self._convert_dict_sentinels(d)
            else:
                sources = csr[i]
                if len(sources) > 0:
                    result[n] = set(sources.tolist())
        return result

    def _has_forward_edges(self, n: int) -> bool:
        """Check if node n has any forward edges."""
        if self._is_mmap:
            i = n - 1
            if i < 0 or i >= len(self._data):
                return False
            return self._data.indptr[i] < self._data.indptr[i + 1]
        return n in self._data

    def _has_inverse_edges(self, n: int) -> bool:
        """Check if node n has any inverse edges."""
        if self._is_mmap:
            if self._dataInv is None:
                return False
            i = n - 1
            if i < 0 or i >= len(self._dataInv):
                return False
            return self._dataInv.indptr[i] < self._dataInv.indptr[i + 1]
        return n in self._dataInv

    def _get_forward_edges(self, n: int) -> set[int] | dict[int, Any] | Any | None:
        """Get raw forward edges for node n.

        Returns:
            For edges without values: set or numpy array of target nodes
            For edges with values: dict with sentinel values converted to None
        """
        if self._is_mmap:
            i = n - 1
            if i < 0 or i >= len(self._data):
                return None
            if self._data.indptr[i] == self._data.indptr[i + 1]:
                return None
            if isinstance(self._data, CSRArrayWithValues):
                indices, values = self._data[i]
                result = dict(zip(indices, values))
                # Convert sentinel values back to None
                return self._convert_dict_sentinels(result)
            else:
                return self._data[i]
        return self._data.get(n)

    def _get_inverse_edges(self, n: int) -> set[int] | dict[int, Any] | Any | None:
        """Get raw inverse edges for node n.

        Returns:
            For edges without values: set or numpy array of source nodes
            For edges with values: dict with sentinel values converted to None
        """
        if self._is_mmap:
            if self._dataInv is None:
                return None
            i = n - 1
            if i < 0 or i >= len(self._dataInv):
                return None
            if self._dataInv.indptr[i] == self._dataInv.indptr[i + 1]:
                return None
            if isinstance(self._dataInv, CSRArrayWithValues):
                indices, values = self._dataInv[i]
                result = dict(zip(indices, values))
                # Convert sentinel values back to None
                return self._convert_dict_sentinels(result)
            else:
                return self._dataInv[i]
        return self._dataInv.get(n)

    def items(self) -> Iterator[tuple[int, set[int] | dict[int, Any]]]:
        """A generator that yields the items of the feature, seen as a mapping.

        This gives you a rather efficient way to iterate over
        just the feature data.

        If you need this repeatedly, or you need the whole dictionary,
        you can store the result as follows:

           data = dict(E.fff.items())

        """
        if self._is_mmap:
            # Iterate over CSR data directly without full materialization
            csr = self._data
            for i in range(len(csr)):
                if csr.indptr[i] < csr.indptr[i + 1]:
                    n = i + 1  # 0-indexed CSR to 1-indexed nodes
                    if isinstance(csr, CSRArrayWithValues):
                        indices, values = csr[i]
                        d = dict(zip(indices, values))
                        yield (n, self._convert_dict_sentinels(d))
                    else:
                        yield (n, set(csr[i]))
        else:
            yield from self._data.items()

    def f(self, n: int) -> tuple[int, ...] | tuple[tuple[int, Any], ...]:
        """Get outgoing edges *from* a node.

        The edges are those pairs of nodes specified in the feature data,
        whose first node is the `n`.

        Parameters
        ----------
        node: integer
            The node **from** which the edges in question start.

        Returns
        -------
        set | tuple
            The nodes reached by the edges **from** a certain node.
            The members of the result are just nodes, if this feature does not
            assign values to edges.
            Otherwise the members are tuples of the destination node and the
            value that the feature assigns to this pair of nodes.

            If there are no edges from the node, the empty tuple is returned,
            rather than `None`.
        """
        edges = self._get_forward_edges(n)
        if edges is None:
            return ()

        # Handle empty edges (for legacy sets/dicts that might be empty)
        if hasattr(edges, '__len__') and len(edges) == 0:
            return ()

        Crank = self.api.C.rank.data
        if self.doValues:
            # edges is a dict for both backends
            return tuple(sorted(edges.items(), key=lambda mv: Crank[mv[0] - 1]))
        else:
            # For mmap backend: edges is tuple
            # For legacy backend: edges is set
            return tuple(sorted(edges, key=lambda m: Crank[m - 1]))

    def t(self, n: int) -> tuple[int, ...] | tuple[tuple[int, Any], ...]:
        """Get incoming edges *to* a node.

        The edges are those pairs of nodes specified in the feature data,
        whose second node is the `n`.

        Parameters
        ----------
        node: integer
            The node **to** which the edges in question connect.

        Returns
        -------
        set | tuple
            The nodes where the edges **to** a certain node start.
            The members of the result are just nodes, if this feature does not
            assign values to edges.
            Otherwise the members are tuples of the start node and the
            value that the feature assigns to this pair of nodes.

            If there are no edges to the node, the empty tuple is returned,
            rather than `None`.
        """
        edges = self._get_inverse_edges(n)
        if edges is None:
            return ()

        # Handle empty edges (for legacy sets/dicts that might be empty)
        if hasattr(edges, '__len__') and len(edges) == 0:
            return ()

        Crank = self.api.C.rank.data
        if self.doValues:
            # edges is a dict for both backends
            return tuple(sorted(edges.items(), key=lambda mv: Crank[mv[0] - 1]))
        else:
            # For mmap backend: edges is tuple
            # For legacy backend: edges is set
            return tuple(sorted(edges, key=lambda m: Crank[m - 1]))

    def b(self, n: int) -> tuple[int, ...] | tuple[tuple[int, Any], ...]:
        """Query *both* incoming edges to, and outgoing edges from a node.

        The edges are those pairs of nodes specified in the feature data,
        whose first or second node is the `n`.

        Parameters
        ----------
        node: integer
            The node **from** which the edges in question start or
            **to** which the edges in question connect.

        Returns
        -------
        set | dict
            The nodes where the edges **to** a certain node start.
            The members of the result are just nodes, if this feature does not
            assign values to edges.
            Otherwise the members are tuples of the start node and the
            value that the feature assigns to this pair of nodes.

            If there are no edges to the node, the empty tuple is returned,
            rather than `None`.

        Notes
        -----
        !!! hint "symmetric closure"
            This method gives the *symmetric closure* of a set of edges:
            if there is an edge between `n` and `m`, this method will deliver
            its value, no matter the direction of the edge.

        !!! example "symmetric edges"
            Some edge sets are semantically symmetric, for example *similarity*.
            If `n` is similar to `m`, then `m` is similar to `n`.

            But if you store such an edge feature completely,
            half of the data is redundant.
            By virtue of this method you do not have to do that, you only need to store
            one of the edges between `n` and `m` (it does not matter which one),
            and `E.fff.b(n)` will nevertheless produce the complete results.

        !!! caution "conflicting values"
            If your set of edges is not symmetric, and edges carry values, it might
            very well be the case that edges between the same pair of nodes carry
            different values for the two directions.

            In that case, this method gives precedence to the edges that
            *depart* from the node to those that go *to* the node.

        !!! example "conflicting values"
            Suppose we have

                n == value=4 ==> m
                m == value=6 ==> n

            then

                E.b(n) = (m, 4)
                E.b(m) = (n, 6)

        """
        has_forward = self._has_forward_edges(n)
        has_inverse = self._has_inverse_edges(n)

        if not has_forward and not has_inverse:
            return ()

        Crank = self.api.C.rank.data

        if self.doValues:
            result = {}
            # Inverse edges first, then forward edges (forward takes precedence)
            inv_edges = self._get_inverse_edges(n)
            if inv_edges:
                result.update(inv_edges.items())
            fwd_edges = self._get_forward_edges(n)
            if fwd_edges:
                result.update(fwd_edges.items())
            return tuple(sorted(result.items(), key=lambda mv: Crank[mv[0] - 1]))
        else:
            result = set()
            inv_edges = self._get_inverse_edges(n)
            if inv_edges is not None:
                if self._is_mmap:
                    result |= set(inv_edges.tolist())
                else:
                    result |= inv_edges
            fwd_edges = self._get_forward_edges(n)
            if fwd_edges is not None:
                if self._is_mmap:
                    result |= set(fwd_edges.tolist())
                else:
                    result |= fwd_edges
            return tuple(sorted(result, key=lambda m: Crank[m - 1]))

    def freqList(
        self, nodeTypesFrom: set[str] | None = None, nodeTypesTo: set[str] | None = None
    ) -> tuple[tuple[Any, int], ...] | int:
        """Frequency list of the values of this feature.

        Inspect the values of this feature and see how often they occur.

        If the feature does not assign values, return the number of node pairs
        in this edge.

        If the edge feature does have values, inspect them and see
        how often they occur.
        The result is a list of pairs `(value, frequency)`, ordered by `frequency`,
        highest frequencies first.

        Parameters
        ----------
        nodeTypesFrom: set of string, optional None
            If you pass a set of node types here, only the values for edges
            that start *from* a node with such a type will be counted.
        nodeTypesTo: set of string, optional None
            If you pass a set of node types here, only the values for edges
            that go *to* a node with such a type will be counted.

        Returns
        -------
        tuple of 2-tuple
            A tuple of `(value, frequency)`, items, ordered by `frequency`,
            highest frequencies first.

        """
        if nodeTypesFrom is None and nodeTypesTo is None:
            if self.doValues:
                fql = collections.Counter()
                for n, vals in self.items():
                    for val in vals.values():
                        fql[val] += 1
                return tuple(sorted(fql.items(), key=lambda x: (-x[1], x[0])))
            else:
                fql = 0
                for n, ms in self.items():
                    fql += len(ms)
                return fql
        else:
            fOtype = self.api.F.otype.v
            if self.doValues:
                fql = collections.Counter()
                for n, vals in self.items():
                    if nodeTypesFrom is None or fOtype(n) in nodeTypesFrom:
                        for m, val in vals.items():
                            if nodeTypesTo is None or fOtype(m) in nodeTypesTo:
                                fql[val] += 1
                return tuple(sorted(fql.items(), key=lambda x: (-x[1], x[0])))
            else:
                fql = 0
                for n, ms in self.items():
                    if nodeTypesFrom is None or fOtype(n) in nodeTypesFrom:
                        for m in ms:
                            if nodeTypesTo is None or fOtype(m) in nodeTypesTo:
                                fql += 1
                return fql
