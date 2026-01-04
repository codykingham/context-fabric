"""
# Access to `otype` feature data.

In general, features are stored as dictionaries, but this specific feature
has an optimised representation. Since it is a large feature and present
in any TF dataset, this pays off.

Supports two backends:
- Legacy tuple format: data = (type_tuple, maxSlot, maxNode, slotType)
- Mmap numpy array: data = numpy uint8/uint16 array, with type_list parameter

"""

import numpy as np


class OtypeFeature:
    def __init__(self, api, metaData, data, type_list=None):
        """Initialize OtypeFeature with either legacy or mmap backend.

        Parameters
        ----------
        api : object
            The API object
        metaData : dict
            Feature metadata
        data : tuple or np.ndarray
            Either:
            - Legacy: tuple (type_tuple, maxSlot, maxNode, slotType)
            - Mmap: numpy uint8/uint16 array of type indices for non-slot nodes
        type_list : list, optional
            When using mmap backend, maps type indices to type strings.
            Required when data is a numpy array.
        """
        self.api = api
        self.meta = metaData
        """Metadata of the feature.

        This is the information found in the lines starting with `@`
        in the `.tf` feature file.
        """

        # Detect backend type based on data format
        self._is_mmap = isinstance(data, np.ndarray)

        if self._is_mmap:
            # Mmap backend: data is numpy array of type indices
            self._data = data
            self._type_list = type_list
            # maxSlot, maxNode, and slotType must be provided via type_list metadata
            # or extracted from the API later; for now we expect them in type_list dict
            if isinstance(type_list, dict):
                self.maxSlot = type_list['maxSlot']
                self.maxNode = type_list['maxNode']
                self.slotType = type_list['slotType']
                self._type_list = type_list['types']
            else:
                # type_list is a simple list, metadata comes from elsewhere
                self._type_list = type_list
                self.maxSlot = None
                self.maxNode = None
                self.slotType = None
        else:
            # Legacy tuple format
            self._data = data[0]
            self.maxSlot = data[1]
            """Last slot node in the corpus."""

            self.maxNode = data[2]
            """Last node node.in the corpus."""

            self.slotType = data[3]
            """The name of the slot type."""

            self._type_list = None

        self.all = None
        """List of all node types from big to small."""

        self.support = {}
        """Support dict for s() method: type -> (min_node, max_node)."""

    @property
    def data(self):
        """Legacy access to raw type data.

        For legacy backend, returns the type tuple.
        For mmap backend, returns the numpy array.
        """
        return self._data

    def items(self):
        """As in `tf.core.nodefeature.NodeFeature.items`."""

        slotType = self.slotType
        maxSlot = self.maxSlot

        for n in range(1, maxSlot + 1):
            yield (n, slotType)

        maxNode = self.maxNode
        shift = maxSlot + 1

        if self._is_mmap:
            # Mmap backend: look up type string via index
            type_list = self._type_list
            data = self._data
            for n in range(maxSlot + 1, maxNode + 1):
                yield (n, type_list[data[n - shift]])
        else:
            # Legacy backend: direct string access
            data = self._data
            for n in range(maxSlot + 1, maxNode + 1):
                yield (n, data[n - shift])

    def v(self, n):
        """Get the node type of a node.

        Parameters
        ----------
        n: integer
            The node in question

        Returns
        -------
        string
            The node type of that node. All nodes have a node type, and it is
            always a string.
        """

        if n == 0:
            return None
        if n < self.maxSlot + 1:
            return self.slotType
        m = n - self.maxSlot
        if m <= len(self._data):
            if self._is_mmap:
                return self._type_list[self._data[m - 1]]
            else:
                return self._data[m - 1]
        return None

    def s(self, val):
        """Query all nodes having a specified node type.

        This is an other way to walk through nodes than using
        `tf.core.nodes.Nodes.walk`.

        Parameters
        ----------
        val: integer | string
            The node type that all resulting nodes have.

        Returns
        -------
        tuple of integer
            All nodes that have this node type, sorted in the canonical order.
            (`tf.core.nodes`)
        """

        # NB: the support attribute has been added by pre-computing __levels__
        if val in self.support:
            (b, e) = self.support[val]
            # N.B. for a long time we delivered range(b, e + 1)
            # thereby forgetting to sort these nodes canonically.
            # Because we cannot assume that nodes of non-slot types are already
            # canonically sorted.
            # That's a pity, because now we need more memory!
            Crank = self.api.C.rank.data
            return tuple(
                sorted(
                    range(b, e + 1),
                    key=lambda n: Crank[n - 1],
                )
            )
        else:
            return ()

    def sInterval(self, val):
        """The interval of nodes having a specified node type.

        The nodes are organized in intervals of nodes with the same type.
        For each type there is only one such interval.
        The first interval, `1:maxSlot + 1` is reserved for the slot type.

        Parameters
        ----------
        val: integer | string
            The node type in question.

        Returns
        -------
        2-tuple of integer
            The start and end node of the interval of nodes with this type.
        """

        # NB: the support attribute has been added by pre-computing __levels__
        if val in self.support:
            return self.support[val]
        else:
            return ()
