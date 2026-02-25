import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rgpycrumbs._aux import (
    _DEPENDENCY_MAP,
    _get_dep_cache_dir,
    _has_cuda,
    _resolve_pip_spec,
    ensure_import,
    lazy_import,
)

pytestmark = pytest.mark.pure


# ---------------------------------------------------------------------------
# _has_cuda
# ---------------------------------------------------------------------------
class TestHasCuda:
    def setup_method(self):
        import rgpycrumbs._aux as aux

        aux._cuda_available = None  # reset cache

    def teardown_method(self):
        import rgpycrumbs._aux as aux

        aux._cuda_available = None

    @patch("shutil.which", return_value=None)
    def test_no_nvidia_smi(self, _mock_which):
        assert _has_cuda() is False

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_nvidia_smi_success(self, _mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert _has_cuda() is True

    @patch("subprocess.run", side_effect=OSError("no device"))
    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_nvidia_smi_oserror(self, _mock_which, _mock_run):
        assert _has_cuda() is False

    @patch("shutil.which", return_value=None)
    def test_result_cached(self, mock_which):
        _has_cuda()
        _has_cuda()
        # shutil.which called once; second call uses cache
        mock_which.assert_called_once()


# ---------------------------------------------------------------------------
# _get_dep_cache_dir
# ---------------------------------------------------------------------------
class TestGetDepCacheDir:
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = _get_dep_cache_dir()
        assert result == Path.home() / ".cache" / "rgpycrumbs" / "deps"

    def test_xdg_override(self, monkeypatch, tmp_path):
        xdg = str(tmp_path / "xdg")
        monkeypatch.setenv("XDG_CACHE_HOME", xdg)
        result = _get_dep_cache_dir()
        assert result == Path(xdg) / "rgpycrumbs" / "deps"


# ---------------------------------------------------------------------------
# _resolve_pip_spec
# ---------------------------------------------------------------------------
class TestResolvePipSpec:
    def setup_method(self):
        import rgpycrumbs._aux as aux

        aux._cuda_available = None

    def teardown_method(self):
        import rgpycrumbs._aux as aux

        aux._cuda_available = None

    @patch("rgpycrumbs._aux._has_cuda", return_value=False)
    def test_jax_cpu_override(self, _mock_cuda):
        spec = _resolve_pip_spec("jax")
        assert "cpu" in spec

    @patch("rgpycrumbs._aux._has_cuda", return_value=True)
    def test_jax_cuda_no_override(self, _mock_cuda):
        spec = _resolve_pip_spec("jax")
        assert spec == "jax>=0.4"

    @patch("rgpycrumbs._aux._has_cuda", return_value=False)
    def test_scipy_no_override(self, _mock_cuda):
        spec = _resolve_pip_spec("scipy")
        assert spec == "scipy>=1.11"


# ---------------------------------------------------------------------------
# ensure_import
# ---------------------------------------------------------------------------
class TestEnsureImport:
    @patch("importlib.import_module")
    def test_already_available(self, mock_import):
        """Module in current env: no subprocess, no uv."""
        fake_mod = types.ModuleType("jax")
        mock_import.return_value = fake_mod

        result = ensure_import("jax")

        assert result is fake_mod
        mock_import.assert_called_once_with("jax")

    @patch("importlib.import_module")
    def test_from_parent_env(self, mock_import, monkeypatch):
        """Falls back to parent site-packages."""
        parent_path = "/parent/site-packages"
        monkeypatch.setenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", parent_path)
        monkeypatch.setattr(sys, "path", ["/local/lib"])

        fake_mod = types.ModuleType("jax")
        # 1st call (ensure_import step 1) fails
        # 2nd call (step 1 inside _import_from_parent_env) fails
        # 3rd call (step 2 inside _import_from_parent_env with parent path) succeeds
        mock_import.side_effect = [
            ImportError("not here"),
            ImportError("not local"),
            fake_mod,
        ]

        result = ensure_import("jax")
        assert result is fake_mod

    @patch("importlib.import_module")
    def test_from_cache_dir(self, mock_import, monkeypatch, tmp_path):
        """Finds module in uv cache directory."""
        monkeypatch.delenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", raising=False)
        monkeypatch.setattr(sys, "path", ["/local/lib"])

        cache_dir = tmp_path / "rgpycrumbs" / "deps"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr("rgpycrumbs._aux._get_dep_cache_dir", lambda: cache_dir)

        fake_mod = types.ModuleType("jax")
        # step 1 fails, _import_from_parent_env returns None (no env var),
        # step 3 (cache) succeeds
        call_count = 0

        def side_effect(_name):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ImportError("nope")
            return fake_mod

        mock_import.side_effect = side_effect

        result = ensure_import("jax")
        assert result is fake_mod
        assert str(cache_dir) in sys.path

    def test_uv_install_triggered(self, monkeypatch, tmp_path):
        """With RGPYCRUMBS_AUTO_DEPS=1, triggers uv install."""
        monkeypatch.setenv("RGPYCRUMBS_AUTO_DEPS", "1")
        monkeypatch.delenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", raising=False)

        cache_dir = tmp_path / "rgpycrumbs" / "deps"
        monkeypatch.setattr("rgpycrumbs._aux._get_dep_cache_dir", lambda: cache_dir)

        uv_called_with = []
        original_import = importlib.import_module

        def mock_uv_install(spec, target):
            uv_called_with.append((spec, target))
            # Simulate uv creating the cache dir and "installing" scipy
            target.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr("rgpycrumbs._aux._uv_install", mock_uv_install)

        fake_mod = types.ModuleType("scipy")
        installed = False

        def mock_import(name):
            if name == "scipy" and not installed:
                raise ImportError("not installed")
            if name == "scipy" and installed:
                return fake_mod
            return original_import(name)

        # First, the module is missing; after uv_install, it appears.
        # We toggle `installed` once _uv_install has been called.
        def uv_install_and_toggle(spec, target):
            nonlocal installed
            uv_called_with.append((spec, target))
            target.mkdir(parents=True, exist_ok=True)
            installed = True

        monkeypatch.setattr("rgpycrumbs._aux._uv_install", uv_install_and_toggle)
        monkeypatch.setattr(importlib, "import_module", mock_import)

        result = ensure_import("scipy")
        assert result is fake_mod
        assert len(uv_called_with) == 1
        assert "scipy" in uv_called_with[0][0]

    @patch("importlib.import_module")
    def test_no_auto_deps_raises(self, mock_import, monkeypatch):
        """Without RGPYCRUMBS_AUTO_DEPS, raises ImportError with message."""
        monkeypatch.delenv("RGPYCRUMBS_AUTO_DEPS", raising=False)
        monkeypatch.delenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", raising=False)
        monkeypatch.setattr(sys, "path", ["/local/lib"])
        mock_import.side_effect = ImportError("nope")

        with pytest.raises(ImportError, match="pip install rgpycrumbs"):
            ensure_import("jax")

    @patch("importlib.import_module")
    def test_unknown_module_message(self, mock_import, monkeypatch):
        """Unknown modules get a conda/pixi suggestion."""
        monkeypatch.delenv("RGPYCRUMBS_AUTO_DEPS", raising=False)
        monkeypatch.delenv("RGPYCRUMBS_PARENT_SITE_PACKAGES", raising=False)
        monkeypatch.setattr(sys, "path", ["/local/lib"])
        mock_import.side_effect = ImportError("nope")

        with pytest.raises(ImportError, match="pixi"):
            ensure_import("ira_mod")


# ---------------------------------------------------------------------------
# lazy_import
# ---------------------------------------------------------------------------
class TestLazyModule:
    @patch("rgpycrumbs._aux.ensure_import")
    def test_defers_import(self, mock_ensure):
        """Proxy does not call ensure_import at construction time."""
        proxy = lazy_import("jax")
        mock_ensure.assert_not_called()
        assert "unresolved" in repr(proxy)

    @patch("rgpycrumbs._aux.ensure_import")
    def test_resolves_on_attr(self, mock_ensure):
        """First attribute access triggers ensure_import."""
        fake_mod = types.ModuleType("jax")
        fake_mod.numpy = "jnp"
        mock_ensure.return_value = fake_mod

        proxy = lazy_import("jax")
        result = proxy.numpy

        assert result == "jnp"
        mock_ensure.assert_called_once_with("jax")

    @patch("rgpycrumbs._aux.ensure_import")
    def test_caches_resolution(self, mock_ensure):
        """Subsequent accesses reuse the resolved module."""
        fake_mod = types.ModuleType("jax")
        fake_mod.a = 1
        fake_mod.b = 2
        mock_ensure.return_value = fake_mod

        proxy = lazy_import("jax")
        _ = proxy.a
        _ = proxy.b

        mock_ensure.assert_called_once()


# ---------------------------------------------------------------------------
# Dependency map consistency
# ---------------------------------------------------------------------------
class TestDependencyMap:
    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="tomllib requires Python 3.11+"
    )
    def test_all_extras_match_pyproject(self):
        """Every extra name in _DEPENDENCY_MAP must exist in pyproject.toml."""
        import tomllib

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        extras = set(data["project"]["optional-dependencies"].keys())
        for _spec, extra_name in _DEPENDENCY_MAP.values():
            assert extra_name in extras, (
                f"Extra '{extra_name}' from _DEPENDENCY_MAP "
                f"not found in pyproject.toml (available: {extras})"
            )

    def test_map_has_expected_entries(self):
        assert "jax" in _DEPENDENCY_MAP
        assert "scipy" in _DEPENDENCY_MAP
        assert "ase" in _DEPENDENCY_MAP
        # Conda-only deps must NOT be in the map
        assert "ira_mod" not in _DEPENDENCY_MAP
        assert "tblite" not in _DEPENDENCY_MAP
        assert "ovito" not in _DEPENDENCY_MAP
