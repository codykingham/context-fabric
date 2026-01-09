"""Integration tests for embedding preload opt-out mechanism.

Tests that embedding preloading can be controlled via the CF_EMBEDDING_CACHE
environment variable.
"""

import pytest


class TestPreloadDefault:
    """Tests that preloading is enabled by default."""

    def test_levup_preloaded_by_default(self, loaded_api):
        """levUp should be preloaded by default after loading."""
        # With default settings (CF_EMBEDDING_CACHE=on), levUp should be cached
        assert loaded_api.C.levUp.is_cached, "levUp should be preloaded by default"

    def test_levdown_preloaded_by_default(self, loaded_api):
        """levDown should be preloaded by default after loading."""
        assert loaded_api.C.levDown.is_cached, "levDown should be preloaded by default"


class TestPreloadOptOut:
    """Tests that preloading can be disabled."""

    def test_preload_disabled_via_module_var(self, mini_corpus_path):
        """Embedding preload can be disabled by setting _EMBEDDING_CACHE_MODE='off'."""
        import cfabric.storage.csr as csr_module

        # Save original value
        original_mode = csr_module._EMBEDDING_CACHE_MODE

        try:
            # Disable preloading
            csr_module._EMBEDDING_CACHE_MODE = 'off'

            from cfabric.core.fabric import Fabric

            cf = Fabric(locations=mini_corpus_path, silent="deep")
            api = cf.loadAll(silent="deep")

            assert not api.C.levUp.is_cached, \
                "levUp should NOT be preloaded when _EMBEDDING_CACHE_MODE='off'"
            assert not api.C.levDown.is_cached, \
                "levDown should NOT be preloaded when _EMBEDDING_CACHE_MODE='off'"

        finally:
            # Restore original value
            csr_module._EMBEDDING_CACHE_MODE = original_mode


class TestPreloadManualOverride:
    """Tests that manual preloading works regardless of auto-preload setting."""

    def test_manual_preload_after_optout(self, mini_corpus_path):
        """Manual preload() should work even when auto-preload is disabled."""
        import cfabric.storage.csr as csr_module

        original_mode = csr_module._EMBEDDING_CACHE_MODE

        try:
            # Disable auto-preloading
            csr_module._EMBEDDING_CACHE_MODE = 'off'

            from cfabric.core.fabric import Fabric

            cf = Fabric(locations=mini_corpus_path, silent="deep")
            api = cf.loadAll(silent="deep")

            # Initially not cached
            assert not api.C.levUp.is_cached

            # Manual preload should work
            api.C.levUp.preload()
            api.C.levDown.preload()

            assert api.C.levUp.is_cached, "Manual preload should work"
            assert api.C.levDown.is_cached, "Manual preload should work"

        finally:
            csr_module._EMBEDDING_CACHE_MODE = original_mode

    def test_release_after_autopreload(self, loaded_api):
        """release() should free memory after auto-preload."""
        # Initially cached due to auto-preload
        assert loaded_api.C.levUp.is_cached

        # Release should work
        loaded_api.C.levUp.release()
        loaded_api.C.levDown.release()

        assert not loaded_api.C.levUp.is_cached, "release() should free cache"
        assert not loaded_api.C.levDown.is_cached, "release() should free cache"


class TestEnvVarModuleVariable:
    """Tests for the _EMBEDDING_CACHE_MODE module variable behavior."""

    def test_default_value_is_on(self):
        """Default value of _EMBEDDING_CACHE_MODE should be 'on'."""
        import cfabric.storage.csr as csr_module

        # Note: If env var is set during test run, this may not be 'on'
        # But at module init with no env var, default should be 'on'
        # We verify the code logic rather than runtime state
        import os
        if "CF_EMBEDDING_CACHE" not in os.environ:
            assert csr_module._EMBEDDING_CACHE_MODE == "on", \
                "Default _EMBEDDING_CACHE_MODE should be 'on'"

    def test_module_var_controls_preload(self, mini_corpus_path):
        """_EMBEDDING_CACHE_MODE variable directly controls preload behavior."""
        import cfabric.storage.csr as csr_module

        original_mode = csr_module._EMBEDDING_CACHE_MODE

        try:
            # Test with 'off'
            csr_module._EMBEDDING_CACHE_MODE = 'off'

            from cfabric.core.fabric import Fabric

            cf = Fabric(locations=mini_corpus_path, silent="deep")
            api = cf.loadAll(silent="deep")

            assert not api.C.levUp.is_cached

        finally:
            csr_module._EMBEDDING_CACHE_MODE = original_mode

    def test_case_sensitivity_of_off(self, mini_corpus_path):
        """Only lowercase 'off' should disable preloading."""
        import cfabric.storage.csr as csr_module

        original_mode = csr_module._EMBEDDING_CACHE_MODE

        try:
            # 'OFF' (uppercase) should NOT disable because code uses .lower()
            # But since we're setting directly, 'OFF' != 'off'
            csr_module._EMBEDDING_CACHE_MODE = 'OFF'

            from cfabric.core.fabric import Fabric

            cf = Fabric(locations=mini_corpus_path, silent="deep")
            api = cf.loadAll(silent="deep")

            # 'OFF' is not equal to 'off', so preloading should happen
            assert api.C.levUp.is_cached, \
                "Only exact 'off' should disable preloading (env var is lowercased at read)"

        finally:
            csr_module._EMBEDDING_CACHE_MODE = original_mode
