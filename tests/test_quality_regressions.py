"""Regression tests, one or more per fixed defect."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import pytest

pytestmark = pytest.mark.pure


@pytest.mark.parametrize(
    ("raw", "normalized", "enabled"),
    [
        ("true", "1", True),
        ("YES", "1", True),
        ("on", "1", True),
        ("1", "1", True),
        ("false", "0", False),
        ("No", "0", False),
        ("off", "0", False),
        ("0", "0", False),
    ],
)
def test_auto_deps_environment_flags_share_boolean_semantics(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    normalized: str,
    enabled: bool,
) -> None:
    """Configuration and importer accept the same documented flag values."""
    from rgpycrumbs._aux import _auto_deps_enabled
    from rgpycrumbs.config import resolve_auto_deps_default

    monkeypatch.delenv("RGPKGS_AUTO_DEPS", raising=False)
    monkeypatch.setenv("RGPYCRUMBS_AUTO_DEPS", raw)

    assert resolve_auto_deps_default() == normalized
    assert _auto_deps_enabled() is enabled


def test_unrecognized_auto_deps_flag_counts_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value outside the boolean spellings falls through to the next source."""
    from rgpycrumbs._aux import _auto_deps_enabled
    from rgpycrumbs.config import RgpycrumbsConfig, resolve_auto_deps_default

    monkeypatch.setenv("RGPKGS_AUTO_DEPS", "maybe")
    monkeypatch.setenv("RGPYCRUMBS_AUTO_DEPS", "maybe")

    assert resolve_auto_deps_default(config=RgpycrumbsConfig(auto_deps=False)) == "0"
    assert resolve_auto_deps_default(config=RgpycrumbsConfig()) == "1"
    assert _auto_deps_enabled() is False

    monkeypatch.setenv("RGPYCRUMBS_AUTO_DEPS", "on")

    assert resolve_auto_deps_default(config=RgpycrumbsConfig(auto_deps=False)) == "1"
    assert _auto_deps_enabled() is True


def test_version_falls_back_to_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the generated module, the installed metadata names the version."""
    import rgpycrumbs

    monkeypatch.setitem(sys.modules, "rgpycrumbs._version", None)
    try:
        reloaded = importlib.reload(rgpycrumbs)
        assert reloaded.__version__ == importlib.metadata.version("rgpycrumbs")
    finally:
        monkeypatch.undo()
        importlib.reload(rgpycrumbs)
