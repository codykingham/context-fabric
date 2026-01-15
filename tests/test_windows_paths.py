"""Tests for Windows path handling in expandDir()."""
from unittest.mock import patch, MagicMock
import pytest


class TestExpandDirWindowsPaths:
    """Test expandDir() handles Windows absolute paths correctly."""

    def test_windows_absolute_path_not_modified(self):
        """Windows absolute path should not have curDir prepended.

        Regression test for issue where Windows paths with drive letters
        were incorrectly treated as relative paths.
        """
        from cfabric.utils.files import expandDir

        # Create mock object with curDir set
        obj = MagicMock()
        obj.curDir = "c:/projects/corpora"
        obj.homeDir = "c:/Users/testuser"
        obj.parentDir = "c:/projects"

        # Simulate Windows: patch os.path.isabs to recognize Windows paths
        def windows_isabs(path):
            if path.startswith("/"):
                return True
            if len(path) >= 2 and path[1] == ":":
                return True
            return False

        with patch("cfabric.utils.files.os.path.isabs", side_effect=windows_isabs):
            result = expandDir(obj, "c:/data/my-corpus")

        # Should NOT prepend curDir - path is already absolute
        assert "c:/projects/corpora/c:" not in result.lower(), \
            f"Windows absolute path was incorrectly treated as relative: {result}"
        assert result == "c:/data/my-corpus"

    def test_relative_path_gets_curdir_prepended(self):
        """Relative paths should still get curDir prepended."""
        from cfabric.utils.files import expandDir

        obj = MagicMock()
        obj.curDir = "c:/projects/corpora"
        obj.homeDir = "c:/Users/testuser"
        obj.parentDir = "c:/projects"

        def windows_isabs(path):
            if path.startswith("/"):
                return True
            if len(path) >= 2 and path[1] == ":":
                return True
            return False

        with patch("cfabric.utils.files.os.path.isabs", side_effect=windows_isabs):
            result = expandDir(obj, "my-corpus")

        assert result == "c:/projects/corpora/my-corpus"

    def test_unix_absolute_path_not_modified(self):
        """Unix absolute paths should not have curDir prepended."""
        from cfabric.utils.files import expandDir

        obj = MagicMock()
        obj.curDir = "/home/user/corpora"
        obj.homeDir = "/home/user"
        obj.parentDir = "/home/user"

        result = expandDir(obj, "/data/my-corpus")

        assert result == "/data/my-corpus"
        assert not result.startswith("/home/user/corpora/")
