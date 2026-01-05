"""Tests for CSR array utilities."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from cfabric.storage.csr import CSRArray, CSRArrayWithValues


class TestCSRArray:
    """Test CSRArray basic functionality."""

    def test_from_sequences_simple(self):
        """CSRArray can be built from simple sequences."""
        sequences = [[1, 2, 3], [4, 5], [6]]
        csr = CSRArray.from_sequences(sequences)

        assert len(csr) == 3
        assert list(csr[0]) == [1, 2, 3]
        assert list(csr[1]) == [4, 5]
        assert list(csr[2]) == [6]

    def test_empty_rows(self):
        """CSRArray handles empty rows correctly."""
        sequences = [[1], [], [2, 3], []]
        csr = CSRArray.from_sequences(sequences)

        assert len(csr) == 4
        assert list(csr[0]) == [1]
        assert list(csr[1]) == []
        assert list(csr[2]) == [2, 3]
        assert list(csr[3]) == []

    def test_get_as_tuple(self):
        """get_as_tuple returns tuple for API compatibility."""
        sequences = [[1, 2, 3]]
        csr = CSRArray.from_sequences(sequences)

        result = csr.get_as_tuple(0)
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_save_load_roundtrip(self):
        """CSRArray can be saved and loaded."""
        sequences = [[1, 2], [], [3, 4, 5]]
        csr = CSRArray.from_sequences(sequences)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test'
            csr.save(str(path))

            loaded = CSRArray.load(str(path))
            assert len(loaded) == len(csr)
            for i in range(len(csr)):
                assert list(loaded[i]) == list(csr[i])


class TestCSRArrayWithValues:
    """Test CSRArrayWithValues for edges with values."""

    def test_from_dict_of_dicts(self):
        """CSRArrayWithValues can be built from dict of dicts."""
        data = {
            0: {10: 100, 20: 200},
            2: {30: 300},
        }
        csr = CSRArrayWithValues.from_dict_of_dicts(data, num_rows=3)

        indices, values = csr[0]
        assert list(indices) == [10, 20]
        assert list(values) == [100, 200]

        indices, values = csr[1]  # empty row
        assert len(indices) == 0

        indices, values = csr[2]
        assert list(indices) == [30]
        assert list(values) == [300]

    def test_get_as_dict(self):
        """get_as_dict returns dict for API compatibility."""
        data = {0: {10: 100, 20: 200}}
        csr = CSRArrayWithValues.from_dict_of_dicts(data, num_rows=1)

        result = csr.get_as_dict(0)
        assert result == {10: 100, 20: 200}
