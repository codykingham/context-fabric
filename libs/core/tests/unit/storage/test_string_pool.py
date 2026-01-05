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
