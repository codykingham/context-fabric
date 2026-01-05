"""Tests for MmapManager."""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from cfabric.storage.mmap_manager import MmapManager
from cfabric.storage.csr import CSRArray


class TestMmapManager:
    """Test MmapManager for lazy loading."""

    @pytest.fixture
    def cfm_dir(self):
        """Create a minimal .cfm directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfm_path = Path(tmpdir) / '.cfm' / '1'
            cfm_path.mkdir(parents=True)

            # Create meta.json
            meta = {
                'max_slot': 5,
                'max_node': 8,
                'slot_type': 'word',
                'node_types': ['word', 'phrase', 'sentence'],
            }
            with open(cfm_path / 'meta.json', 'w') as f:
                json.dump(meta, f)

            # Create warp directory with sample arrays
            warp_dir = cfm_path / 'warp'
            warp_dir.mkdir()

            otype_arr = np.array([0, 0, 1], dtype='uint8')
            np.save(str(warp_dir / 'otype.npy'), otype_arr)

            yield cfm_path

    def test_meta_properties(self, cfm_dir):
        """MmapManager loads and exposes metadata properties."""
        mgr = MmapManager(cfm_dir)

        assert mgr.max_slot == 5
        assert mgr.max_node == 8
        assert mgr.slot_type == 'word'
        assert 'phrase' in mgr.node_types

    def test_lazy_array_loading(self, cfm_dir):
        """Arrays are loaded lazily."""
        mgr = MmapManager(cfm_dir)

        # Not loaded yet
        assert len(mgr._arrays) == 0

        # Load array
        arr = mgr.get_array('warp', 'otype')

        # Now cached
        assert len(mgr._arrays) == 1
        assert list(arr) == [0, 0, 1]

    def test_exists(self, cfm_dir):
        """exists() checks for meta.json."""
        mgr = MmapManager(cfm_dir)
        assert mgr.exists()

        # Non-existent path
        mgr2 = MmapManager(cfm_dir.parent / 'nonexistent')
        assert not mgr2.exists()

    def test_close(self, cfm_dir):
        """close() releases cached arrays."""
        mgr = MmapManager(cfm_dir)
        mgr.get_array('warp', 'otype')  # Load something

        assert len(mgr._arrays) > 0
        mgr.close()
        assert len(mgr._arrays) == 0
