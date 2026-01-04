"""
# Access to `oslots` feature data.

In general, features are stored as dictionaries, but this specific feature
has an optimised representation. Since it is a large feature and present
in any TF dataset, this pays off.

Supports two backends:
- Legacy tuple format: data = (slots_tuple, maxSlot, maxNode)
- CSR array format: data = CSRArray instance

"""

from .csr import CSRArray


class OslotsFeature:
    def __init__(self, api, metaData, data, maxSlot=None, maxNode=None):
        """Initialize OslotsFeature with either legacy or CSR backend.

        Parameters
        ----------
        api : object
            The API object
        metaData : dict
            Feature metadata
        data : tuple or CSRArray
            Either:
            - Legacy: tuple (slots_tuple, maxSlot, maxNode)
            - CSR: CSRArray instance containing slot data for non-slot nodes
        maxSlot : int, optional
            When using CSR backend, the maximum slot node number.
            Required when data is a CSRArray.
        maxNode : int, optional
            When using CSR backend, the maximum node number.
            Required when data is a CSRArray.
        """
        self.api = api
        self.meta = metaData
        """Metadata of the feature.

        This is the information found in the lines starting with `@`
        in the `.tf` feature file.
        """

        # Detect backend type based on data format
        self._is_mmap = isinstance(data, CSRArray)

        if self._is_mmap:
            # CSR backend
            self._data = data
            self.maxSlot = maxSlot
            self.maxNode = maxNode
        else:
            # Legacy tuple format
            self._data = data[0]
            self.maxSlot = data[1]
            self.maxNode = data[2]

    @property
    def data(self):
        """Legacy access to raw slots data.

        For legacy backend, returns the slots tuple.
        For CSR backend, returns the CSRArray.
        """
        return self._data

    def items(self):
        """A generator that yields the non-slot nodes with their slots."""

        maxSlot = self.maxSlot
        maxNode = self.maxNode
        shift = maxSlot + 1

        if self._is_mmap:
            # CSR backend: use get_as_tuple for API compatibility
            data = self._data
            for n in range(maxSlot + 1, maxNode + 1):
                yield (n, data.get_as_tuple(n - shift))
        else:
            # Legacy backend: direct tuple access
            data = self._data
            for n in range(maxSlot + 1, maxNode + 1):
                yield (n, data[n - shift])

    def s(self, n):
        """Get the slots of a (non-slot) node.

        Parameters
        ----------
        node: integer
            The node whose slots must be retrieved.

        Returns
        -------
        tuple
            The slot nodes of the node in question, in canonical order.
            (`tf.core.nodes`)

            For slot nodes `n` it is the tuple `(n,)`.

            All non-slot nodes are linked to at least one slot.
        """

        if n == 0:
            return ()
        if n < self.maxSlot + 1:
            return (n,)
        m = n - self.maxSlot
        if m <= len(self._data):
            if self._is_mmap:
                return self._data.get_as_tuple(m - 1)
            else:
                return self._data[m - 1]
        return ()
