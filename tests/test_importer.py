import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from rgpycrumbs._aux import _import_from_parent_env

pytestmark = pytest.mark.pure


class TestImportFromParentEnv:
    @patch("importlib.import_module")
    def test_import_exists_locally(self, mock_import):
        """
        Scenario: Module exists locally.
        Expected: Returns module immediately, doesn't touch sys.path.
        """
        mock_module = MagicMock()
        mock_import.return_value = mock_module

        result = _import_from_parent_env("my_module")

        assert result == mock_module
        mock_import.assert_called_once_with("my_module")

    @patch("importlib.import_module")
    def test_import_missing_locally_no_env_var(self, mock_import, monkeypatch):
        """
        Scenario: Module missing locally, env var unset.
        Expected: Returns None.
        """
        monkeypatch.delenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", raising=False)
        mock_import.side_effect = ImportError("No module")

        result = _import_from_parent_env("my_module")

        assert result is None
        assert mock_import.call_count == 1

    @patch("importlib.import_module")
    def test_import_missing_locally_exists_in_parent(self, mock_import, monkeypatch):
        """
        Scenario: Module missing locally, found in parent path.
        Expected: Returns module, sys.path modified then restored.
        """
        # Setup environment and paths
        parent_path = "/parent/site-packages"
        monkeypatch.setenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", parent_path)

        # Start with a clean sys.path
        test_sys_path = ["/local/site-packages"]
        monkeypatch.setattr(sys, "path", test_sys_path)

        mock_module = MagicMock()
        # First call fails, second succeeds
        mock_import.side_effect = [ImportError("Fail local"), mock_module]

        result = _import_from_parent_env("my_module")

        assert result == mock_module
        assert mock_import.call_count == 2
        # Verify cleanup
        assert parent_path not in sys.path

    @patch("importlib.import_module")
    def test_import_missing_everywhere(self, mock_import, monkeypatch):
        """
        Scenario: Module missing everywhere.
        Expected: Returns None, sys.path cleaned up.
        """
        monkeypatch.setenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", "/parent/site-packages")
        monkeypatch.setattr(sys, "path", ["/local/site-packages"])

        mock_import.side_effect = [
            ImportError("Fail local"),
            ImportError("Fail parent"),
        ]

        result = _import_from_parent_env("my_module")

        assert result is None
        assert mock_import.call_count == 2
        assert "/parent/site-packages" not in sys.path

    @patch("importlib.import_module")
    def test_sys_path_cleanup_robustness(self, mock_import, monkeypatch):
        """
        Scenario: External modification of sys.path during execution.
        Expected: No crash, clean state.
        """
        monkeypatch.setenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", "/parent/lib")

        # Use a real list we can mutate
        test_path = ["/local/lib"]
        monkeypatch.setattr(sys, "path", test_path)

        def side_effect(*args, **kwargs):
            # If the parent path has been added, remove it prematurely to trigger ValueError in cleanup
            if "/parent/lib" in sys.path:
                sys.path.remove("/parent/lib")
                raise ImportError("Fail parent")
            raise ImportError("Fail local")

        mock_import.side_effect = side_effect

        result = _import_from_parent_env("my_module")

        assert result is None
        assert "/parent/lib" not in sys.path

    @patch("importlib.import_module")
    def test_multiple_paths_handling(self, mock_import, monkeypatch):
        """
        Scenario: Env var has multiple paths.
        Expected: All valid paths temporarily added.
        """
        paths = f"/parent/lib{os.pathsep}/another/lib"
        monkeypatch.setenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", paths)
        monkeypatch.setattr(sys, "path", ["/local/lib"])

        mock_import.side_effect = ImportError("Fail")

        _import_from_parent_env("my_module")

        assert "/parent/lib" not in sys.path
        assert "/another/lib" not in sys.path

    @patch("importlib.import_module")
    def test_path_already_in_sys_path(self, mock_import, monkeypatch):
        """
        Scenario: Parent path is already in sys.path.
        Expected: Don't duplicate it, don't remove it.
        """
        target_path = "/existing/path"
        monkeypatch.setenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", target_path)

        # Path already exists in sys.path
        initial_path = ["/local/lib", target_path]
        monkeypatch.setattr(sys, "path", list(initial_path))

        mock_import.side_effect = ImportError("Fail")

        _import_from_parent_env("my_module")

        assert target_path in sys.path
        assert len(sys.path) == 2
