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

    def test_save_load_roundtrip_int_values(self):
        """CSRArrayWithValues can save/load int values."""
        data = {0: {10: 100, 20: 200}, 2: {30: 300}}
        csr = CSRArrayWithValues.from_dict_of_dicts(data, num_rows=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test'
            csr.save(str(path))
            loaded = CSRArrayWithValues.load(str(path))

            assert loaded.get_as_dict(0) == {10: 100, 20: 200}
            assert loaded.get_as_dict(1) == {}
            assert loaded.get_as_dict(2) == {30: 300}

    def test_save_load_roundtrip_string_values(self):
        """CSRArrayWithValues can save/load string values (mmap-able)."""
        data = {0: {10: 'A0', 20: 'A1'}, 2: {30: 'B0'}}
        csr = CSRArrayWithValues.from_dict_of_dicts(data, num_rows=3, value_dtype=object)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test'
            csr.save(str(path))

            # Must work with mmap_mode='r' (the default for cfm loading)
            loaded = CSRArrayWithValues.load(str(path), mmap_mode='r')

            assert loaded.get_as_dict(0) == {10: 'A0', 20: 'A1'}
            assert loaded.get_as_dict(1) == {}
            assert loaded.get_as_dict(2) == {30: 'B0'}


class TestCSRArrayBatchOperations:
    """Tests for batch/vectorized CSRArray operations."""

    def test_get_all_targets_simple(self):
        """get_all_targets returns union of targets from sources."""
        # Row 0: [10, 20], Row 1: [30], Row 2: [20, 40]
        sequences = [[10, 20], [30], [20, 40]]
        csr = CSRArray.from_sequences(sequences)

        # Sources are 1-indexed
        result = csr.get_all_targets({1, 3})  # rows 0 and 2
        assert result == {10, 20, 40}

    def test_get_all_targets_empty_sources(self):
        """get_all_targets handles empty source set."""
        sequences = [[10, 20], [30]]
        csr = CSRArray.from_sequences(sequences)

        result = csr.get_all_targets(set())
        assert result == set()

    def test_get_all_targets_empty_rows(self):
        """get_all_targets handles sources with empty data."""
        sequences = [[10], [], [20]]
        csr = CSRArray.from_sequences(sequences)

        # Source 2 (row 1) has no data
        result = csr.get_all_targets({1, 2, 3})
        assert result == {10, 20}

    def test_get_all_targets_out_of_bounds(self):
        """get_all_targets ignores out-of-bounds sources."""
        sequences = [[10, 20]]
        csr = CSRArray.from_sequences(sequences)

        result = csr.get_all_targets({1, 100, -1})
        assert result == {10, 20}

    def test_filter_sources_with_targets_in_simple(self):
        """filter_sources_with_targets_in finds matching sources."""
        # Row 0: [10, 20], Row 1: [30], Row 2: [40]
        sequences = [[10, 20], [30], [40]]
        csr = CSRArray.from_sequences(sequences)

        # Sources 1,2,3 (1-indexed), looking for targets 20 or 30
        sources, targets = csr.filter_sources_with_targets_in({1, 2, 3}, {20, 30})

        assert sources == {1, 2}  # Row 0 has 20, Row 1 has 30
        assert targets == {20, 30}

    def test_filter_sources_with_targets_in_no_match(self):
        """filter_sources_with_targets_in returns empty when no matches."""
        sequences = [[10, 20], [30]]
        csr = CSRArray.from_sequences(sequences)

        sources, targets = csr.filter_sources_with_targets_in({1, 2}, {100, 200})
        assert sources == set()
        assert targets == set()

    def test_filter_sources_with_targets_in_empty_sources(self):
        """filter_sources_with_targets_in handles empty source set."""
        sequences = [[10, 20]]
        csr = CSRArray.from_sequences(sequences)

        sources, targets = csr.filter_sources_with_targets_in(set(), {10})
        assert sources == set()
        assert targets == set()

    def test_filter_sources_with_targets_in_empty_targets(self):
        """filter_sources_with_targets_in handles empty target set."""
        sequences = [[10, 20]]
        csr = CSRArray.from_sequences(sequences)

        sources, targets = csr.filter_sources_with_targets_in({1}, set())
        assert sources == set()
        assert targets == set()


class TestCSRArrayPreload:
    """Tests for CSRArray RAM preloading functionality."""

    def test_preload_to_ram(self):
        """preload_to_ram loads data into RAM."""
        sequences = [[1, 2, 3], [4, 5], [6]]
        csr = CSRArray.from_sequences(sequences)

        assert not csr.is_cached
        assert csr.memory_usage_bytes() == 0

        csr.preload_to_ram()

        assert csr.is_cached
        assert csr.memory_usage_bytes() > 0
        # Data should still work correctly
        assert list(csr[0]) == [1, 2, 3]
        assert list(csr[1]) == [4, 5]

    def test_release_cache(self):
        """release_cache frees RAM cache."""
        sequences = [[1, 2], [3, 4]]
        csr = CSRArray.from_sequences(sequences)

        csr.preload_to_ram()
        assert csr.is_cached

        csr.release_cache()
        assert not csr.is_cached
        assert csr.memory_usage_bytes() == 0
        # Data should still work via original arrays
        assert list(csr[0]) == [1, 2]

    def test_preload_idempotent(self):
        """Multiple preload calls don't cause issues."""
        sequences = [[1, 2, 3]]
        csr = CSRArray.from_sequences(sequences)

        csr.preload_to_ram()
        mem1 = csr.memory_usage_bytes()

        csr.preload_to_ram()  # Second call
        mem2 = csr.memory_usage_bytes()

        assert mem1 == mem2  # No extra memory

    def test_filter_with_preload(self):
        """filter_sources_with_targets_in works with preloaded data."""
        sequences = [[10, 20], [30], [20, 40]]
        csr = CSRArray.from_sequences(sequences)

        # Get result without preload
        sources1, targets1 = csr.filter_sources_with_targets_in({1, 2, 3}, {20, 30})

        # Preload and get result again
        csr.preload_to_ram()
        sources2, targets2 = csr.filter_sources_with_targets_in({1, 2, 3}, {20, 30})

        # Results should be identical
        assert sources1 == sources2
        assert targets1 == targets2

