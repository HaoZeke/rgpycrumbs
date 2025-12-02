import os
import sys
from unittest.mock import MagicMock, patch

from rgpycrumbs._aux import _import_from_parent_env


class TestImportFromParentEnv:
    @patch("builtins.__import__")
    def test_import_exists_locally(self, mock_import):
        """
        Scenario: The module exists in the current environment.
        Expected: Returns the module immediately without touching env vars or sys.path.
        """
        # Setup: __import__ succeeds immediately
        mock_module = MagicMock()
        mock_import.return_value = mock_module

        result = _import_from_parent_env("my_module")

        assert result == mock_module
        mock_import.assert_called_once_with("my_module")

    @patch.dict(os.environ, {}, clear=True)
    @patch("builtins.__import__")
    def test_import_missing_locally_no_env_var(self, mock_import):
        """
        Scenario: Module missing locally and RGPYCRUMBS_PARENT_SITE_PACKAGES is unset.
        Expected: Returns None.
        """
        # Setup: first import fails
        mock_import.side_effect = ImportError("No module named 'my_module'")

        result = _import_from_parent_env("my_module")

        assert result is None
        # Should call import once, fail, check env, find nothing, and exit.
        assert mock_import.call_count == 1

    @patch.dict(
        os.environ,
        {"RGPYCRUMBS_PARENT_SITE_PACKAGES": "/parent/site-packages"},
        clear=True,
    )
    @patch("builtins.__import__")
    def test_import_missing_locally_exists_in_parent(self, mock_import):
        """
        Scenario: Module missing locally, but found after adding parent path.
        Expected: Returns the module, sys.path is temporarily modified then restored.
        """
        # Setup:
        # 1. First call fails (local import)
        # 2. Second call succeeds (parent import)
        mock_module = MagicMock()
        mock_import.side_effect = [ImportError("Fail local"), mock_module]

        # Use a list for sys.path so we can verify append/remove
        with patch("sys.path", ["/local/site-packages"]):
            result = _import_from_parent_env("my_module")

            assert result == mock_module
            assert mock_import.call_count == 2

            # Verify clean up happened
            assert "/parent/site-packages" not in sys.path

    @patch.dict(
        os.environ,
        {"RGPYCRUMBS_PARENT_SITE_PACKAGES": "/parent/site-packages"},
        clear=True,
    )
    @patch("builtins.__import__")
    def test_import_missing_everywhere(self, mock_import):
        """
        Scenario: Module missing locally and missing in parent path.
        Expected: Returns None, sys.path is cleaned up.
        """
        # Setup: Both calls fail
        mock_import.side_effect = [
            ImportError("Fail local"),
            ImportError("Fail parent"),
        ]

        with patch("sys.path", ["/local/site-packages"]):
            result = _import_from_parent_env("my_module")

            assert result is None
            assert mock_import.call_count == 2

            # Verify clean up happened despite the second error
            assert "/parent/site-packages" not in sys.path

    @patch.dict(
        os.environ, {"RGPYCRUMBS_PARENT_SITE_PACKAGES": "/parent/lib"}, clear=True
    )
    @patch("builtins.__import__")
    def test_sys_path_cleanup_robustness(self, mock_import):
        """
        Scenario: Ensure sys.path is cleaned up even if something modifies sys.path
        unexpectedly during the import attempt (edge case).
        """
        # First import fails
        mock_import.side_effect = ImportError("Fail local")

        # We start with a clean path
        original_path = ["/local/lib"]

        # We need a real list object to track mutations
        test_path = original_path.copy()

        with patch("sys.path", test_path):
            # FIXED: Added *args and **kwargs to accept the arguments passed by __import__
            def side_effect_second_call(*args, **kwargs):
                if "/parent/lib" in sys.path:
                    # SIMULATE: something removed the path before we could!
                    # This triggers the ValueError in the finally block
                    sys.path.remove("/parent/lib")
                    raise ImportError("Fail parent")
                raise ImportError("Fail local")

            mock_import.side_effect = side_effect_second_call

            result = _import_from_parent_env("my_module")

            assert result is None
            # The function's `finally` block will try to remove "/parent/lib".
            # Since we manually removed it inside the side_effect, the `finally` block
            # will hit a ValueError. We want to ensure the function catches that ValueError
            # and doesn't crash the program.
            assert "/parent/lib" not in sys.path

    @patch.dict(
        os.environ,
        {"RGPYCRUMBS_PARENT_SITE_PACKAGES": f"/parent/lib{os.pathsep}/another/lib"},
        clear=True,
    )
    @patch("builtins.__import__")
    def test_multiple_paths_handling(self, mock_import):
        """
        Scenario: The env var contains multiple paths separated by os.pathsep.
        Expected: All valid paths are added and then removed.
        """
        mock_import.side_effect = ImportError("Fail")

        original_sys_path = ["/local/lib"]

        with patch("sys.path", list(original_sys_path)):
            _import_from_parent_env("my_module")

            # We can't easily check the 'during' state without complex mocking,
            # but we can ensure the logic runs and cleans up multiple paths.
            assert "/parent/lib" not in sys.path
            assert "/another/lib" not in sys.path
            assert sys.path == original_sys_path

    @patch.dict(
        os.environ, {"RGPYCRUMBS_PARENT_SITE_PACKAGES": "/existing/path"}, clear=True
    )
    @patch("builtins.__import__")
    def test_path_already_in_sys_path(self, mock_import):
        """
        Scenario: The path in the env var is ALREADY in sys.path.
        Expected: We should not add it again (duplicate) and consequently should
                  NOT remove it.
        """
        mock_import.side_effect = ImportError("Fail")

        # The parent path is already here
        initial_path = ["/local/lib", "/existing/path"]

        with patch("sys.path", list(initial_path)):
            _import_from_parent_env("my_module")

            # It should still be there (was not removed) because we didn't add it
            assert "/existing/path" in sys.path
            assert len(sys.path) == 2
