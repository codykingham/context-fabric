"""Tests for string pool management."""

import pytest
import tempfile
from pathlib import Path
from cfabric.storage.string_pool import StringPool, IntFeatureArray, MISSING_STR_INDEX


class TestStringPool:
    """Test StringPool for string features."""

    def test_from_dict(self):
        """StringPool can be built from node->string dict."""
        data = {1: 'hello', 3: 'world', 5: 'hello'}
        pool = StringPool.from_dict(data, max_node=6)

        assert pool.get(1) == 'hello'
        assert pool.get(2) is None  # missing
        assert pool.get(3) == 'world'
        assert pool.get(5) == 'hello'  # deduped

    def test_deduplication(self):
        """StringPool deduplicates string values."""
        data = {1: 'same', 2: 'same', 3: 'same'}
        pool = StringPool.from_dict(data, max_node=3)

        # Only one unique string should be stored
        assert len(pool.strings) == 1

    def test_save_load_roundtrip(self):
        """StringPool can be saved and loaded."""
        data = {1: 'hello', 3: 'world'}
        pool = StringPool.from_dict(data, max_node=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test'
            pool.save(str(path))

            loaded = StringPool.load(str(path))
            assert loaded.get(1) == 'hello'
            assert loaded.get(2) is None
            assert loaded.get(3) == 'world'


    def test_out_of_bounds_returns_none(self):
        """StringPool returns None for out-of-bounds nodes."""
        data = {1: 'hello', 2: 'world'}
        pool = StringPool.from_dict(data, max_node=3)

        # Within bounds
        assert pool.get(1) == 'hello'
        assert pool.get(2) == 'world'
        assert pool.get(3) is None  # missing but in bounds

        # Out of bounds - should return None, not crash
        assert pool.get(0) is None
        assert pool.get(-1) is None
        assert pool.get(4) is None
        assert pool.get(1000000) is None


class TestIntFeatureArray:
    """Test IntFeatureArray for integer features."""

    def test_from_dict(self):
        """IntFeatureArray can be built from node->int dict."""
        data = {1: 10, 3: 30, 5: 50}
        arr = IntFeatureArray.from_dict(data, max_node=6)

        assert arr.get(1) == 10
        assert arr.get(2) is None  # missing
        assert arr.get(3) == 30
        assert arr.get(5) == 50

    def test_out_of_bounds_returns_none(self):
        """IntFeatureArray returns None for out-of-bounds nodes."""
        data = {1: 10, 2: 20}
        arr = IntFeatureArray.from_dict(data, max_node=3)

        # Within bounds
        assert arr.get(1) == 10
        assert arr.get(2) == 20
        assert arr.get(3) is None  # missing but in bounds

        # Out of bounds - should return None, not crash
        assert arr.get(0) is None
        assert arr.get(-1) is None
        assert arr.get(4) is None
        assert arr.get(1000000) is None

    def test_save_load_roundtrip(self):
        """IntFeatureArray can be saved and loaded."""
        data = {1: 100, 2: 200}
        arr = IntFeatureArray.from_dict(data, max_node=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.npy'
            arr.save(str(path))

            loaded = IntFeatureArray.load(str(path))
            assert loaded.get(1) == 100
            assert loaded.get(2) == 200
            assert loaded.get(3) is None


class TestStringPoolVectorized:
    """Tests for vectorized filtering operations on StringPool."""

    def test_filter_by_value_returns_matching_nodes(self):
        """filter_by_value should return nodes with specified value."""
        data = {1: 'verb', 2: 'noun', 3: 'verb', 4: 'adj', 5: 'verb'}
        pool = StringPool.from_dict(data, max_node=6)

        # Filter for 'verb' from all nodes
        nodes = [1, 2, 3, 4, 5]
        result = pool.filter_by_value(nodes, 'verb')

        assert set(result) == {1, 3, 5}

    def test_filter_by_value_with_missing_nodes(self):
        """filter_by_value should handle nodes without values."""
        data = {1: 'verb', 3: 'verb', 5: 'noun'}
        pool = StringPool.from_dict(data, max_node=6)

        # Nodes 2, 4, 6 have no values
        nodes = [1, 2, 3, 4, 5, 6]
        result = pool.filter_by_value(nodes, 'verb')

        assert set(result) == {1, 3}

    def test_filter_by_value_nonexistent_value(self):
        """filter_by_value should return empty for nonexistent value."""
        data = {1: 'verb', 2: 'noun'}
        pool = StringPool.from_dict(data, max_node=3)

        result = pool.filter_by_value([1, 2, 3], 'nonexistent')

        assert list(result) == []

    def test_filter_by_value_empty_nodes(self):
        """filter_by_value should handle empty node list."""
        data = {1: 'verb'}
        pool = StringPool.from_dict(data, max_node=2)

        result = pool.filter_by_value([], 'verb')

        assert list(result) == []

    def test_filter_by_values_multiple_values(self):
        """filter_by_values should match any of multiple values."""
        data = {1: 'verb', 2: 'noun', 3: 'adj', 4: 'verb', 5: 'noun'}
        pool = StringPool.from_dict(data, max_node=6)

        nodes = [1, 2, 3, 4, 5]
        result = pool.filter_by_values(nodes, {'verb', 'noun'})

        assert set(result) == {1, 2, 4, 5}

    def test_filter_by_values_single_value(self):
        """filter_by_values with single value should work like filter_by_value."""
        data = {1: 'verb', 2: 'noun', 3: 'verb'}
        pool = StringPool.from_dict(data, max_node=4)

        result = pool.filter_by_values([1, 2, 3], {'verb'})

        assert set(result) == {1, 3}

    def test_filter_by_values_empty_values(self):
        """filter_by_values with empty values set returns empty."""
        data = {1: 'verb', 2: 'noun'}
        pool = StringPool.from_dict(data, max_node=3)

        result = pool.filter_by_values([1, 2], set())

        assert list(result) == []

    def test_get_value_index_returns_index(self):
        """get_value_index should return internal index for value."""
        data = {1: 'hello', 2: 'world'}
        pool = StringPool.from_dict(data, max_node=3)

        # Should return valid indices
        idx_hello = pool.get_value_index('hello')
        idx_world = pool.get_value_index('world')

        assert idx_hello is not None
        assert idx_world is not None
        assert idx_hello != idx_world

    def test_get_value_index_nonexistent(self):
        """get_value_index should return None for nonexistent value."""
        data = {1: 'hello'}
        pool = StringPool.from_dict(data, max_node=2)

        result = pool.get_value_index('nonexistent')

        assert result is None


class TestIntFeatureArrayVectorized:
    """Tests for vectorized filtering operations on IntFeatureArray."""

    def test_filter_by_value_returns_matching_nodes(self):
        """filter_by_value should return nodes with specified value."""
        data = {1: 10, 2: 20, 3: 10, 4: 30, 5: 10}
        arr = IntFeatureArray.from_dict(data, max_node=6)

        nodes = [1, 2, 3, 4, 5]
        result = arr.filter_by_value(nodes, 10)

        assert set(result) == {1, 3, 5}

    def test_filter_by_value_with_missing_nodes(self):
        """filter_by_value should handle nodes without values."""
        data = {1: 10, 3: 10, 5: 20}
        arr = IntFeatureArray.from_dict(data, max_node=6)

        # Nodes 2, 4, 6 have no values
        nodes = [1, 2, 3, 4, 5, 6]
        result = arr.filter_by_value(nodes, 10)

        assert set(result) == {1, 3}

    def test_filter_by_value_nonexistent_value(self):
        """filter_by_value should return empty for nonexistent value."""
        data = {1: 10, 2: 20}
        arr = IntFeatureArray.from_dict(data, max_node=3)

        result = arr.filter_by_value([1, 2, 3], 999)

        assert list(result) == []

    def test_filter_by_values_multiple_values(self):
        """filter_by_values should match any of multiple values."""
        data = {1: 10, 2: 20, 3: 30, 4: 10, 5: 20}
        arr = IntFeatureArray.from_dict(data, max_node=6)

        nodes = [1, 2, 3, 4, 5]
        result = arr.filter_by_values(nodes, {10, 20})

        assert set(result) == {1, 2, 4, 5}

    def test_filter_less_than(self):
        """filter_less_than should return nodes with value < threshold."""
        data = {1: 5, 2: 10, 3: 15, 4: 20, 5: 25}
        arr = IntFeatureArray.from_dict(data, max_node=6)

        result = arr.filter_less_than([1, 2, 3, 4, 5], 15)

        assert set(result) == {1, 2}

    def test_filter_greater_than(self):
        """filter_greater_than should return nodes with value > threshold."""
        data = {1: 5, 2: 10, 3: 15, 4: 20, 5: 25}
        arr = IntFeatureArray.from_dict(data, max_node=6)

        result = arr.filter_greater_than([1, 2, 3, 4, 5], 15)

        assert set(result) == {4, 5}

    def test_filter_has_value(self):
        """filter_has_value should return nodes that have any value."""
        data = {1: 10, 3: 30, 5: 50}
        arr = IntFeatureArray.from_dict(data, max_node=6)

        # Nodes 2, 4, 6 have no values
        result = arr.filter_has_value([1, 2, 3, 4, 5, 6])

        assert set(result) == {1, 3, 5}

    def test_filter_missing_value(self):
        """filter_missing_value should return nodes without values."""
        data = {1: 10, 3: 30, 5: 50}
        arr = IntFeatureArray.from_dict(data, max_node=6)

        result = arr.filter_missing_value([1, 2, 3, 4, 5, 6])

        assert set(result) == {2, 4, 6}
