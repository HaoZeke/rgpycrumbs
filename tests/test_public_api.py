# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Drive shipped public API entry points (not re-implementations)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rgpycrumbs.api import (
    apply_pin_to_spec,
    ensure_import,
    load_config,
    load_pypi_pins,
    normalize_pypi_name,
    pins_to_constraint_lines,
    resolve_lock_path_layered,
    user_config_path,
)

pytestmark = pytest.mark.pure

FIXTURES = Path(__file__).parent / "fixtures"


class TestPublicConfigApi:
    def test_load_config_returns_object_with_pins_map(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.chdir(tmp_path)
        cfg = load_config(cwd=tmp_path)
        assert hasattr(cfg, "package_pins")
        assert hasattr(cfg, "merged_package_pins_normalized")
        pins = cfg.merged_package_pins_normalized()
        assert isinstance(pins, dict)

    def test_user_config_path_under_rgpkgs(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        p = user_config_path()
        assert p.parts[-2:] == ("rgpkgs", "config.toml")

    def test_resolve_lock_path_cli_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        lock = tmp_path / "cli.lock"
        lock.write_text("version = 1\n[[package]]\nname = \"x\"\nversion = \"1\"\n")
        # even empty config
        p = resolve_lock_path_layered(cli_lock=str(lock), cwd=tmp_path)
        assert p is not None
        assert p.resolve() == lock.resolve()


class TestPublicLocksApi:
    def test_load_pypi_pins_from_uv_lock_fixture(self):
        pins = load_pypi_pins(FIXTURES / "minimal_uv.lock")
        assert pins
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"
        assert pins[normalize_pypi_name("jax")] == "0.4.31"

    def test_apply_pin_to_spec_uses_map(self):
        pins = {"jax": "0.4.31"}
        assert apply_pin_to_spec("jax>=0.4", pins) == "jax==0.4.31"
        lines = pins_to_constraint_lines(pins)
        assert lines == ["jax==0.4.31"]


class TestPublicEnsureImport:
    def test_ensure_import_numpy_always_available(self):
        np = ensure_import("numpy")
        assert hasattr(np, "array")
        assert np.array([1, 2]).sum() == 3
