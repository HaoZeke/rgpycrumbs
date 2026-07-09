# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Tests for layered XDG / project TOML config."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rgpycrumbs.config import (
    CONFIG_PATH_ENV,
    find_project_config,
    load_config,
    resolve_auto_deps_default,
    resolve_force_uv,
    resolve_lock_path_layered,
    user_config_path,
)
from rgpycrumbs.locks import LOCK_PATH_ENV, SBOM_PATH_ENV, normalize_pypi_name

pytestmark = pytest.mark.pure


class TestConfigDiscovery:
    def test_user_config_path_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert user_config_path() == tmp_path / "rgpycrumbs" / "config.toml"

    def test_find_project_config_walks_up(self, tmp_path):
        root = tmp_path / "proj"
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        cfg = root / "rgpycrumbs.toml"
        cfg.write_text("[dispatch]\nforce_uv = true\n", encoding="utf-8")
        assert find_project_config(nested) == cfg

    def test_load_merges_global_then_project(self, monkeypatch, tmp_path):
        xdg = tmp_path / "xdg"
        (xdg / "rgpycrumbs").mkdir(parents=True)
        user = xdg / "rgpycrumbs" / "config.toml"
        user.write_text(
            "[dispatch]\nauto_deps = false\nforce_uv = false\n"
            "[pins]\nlock = \"global.lock\"\n"
            "[pins.packages]\njax = \"0.4.0\"\nnumpy = \"1.0.0\"\n",
            encoding="utf-8",
        )
        proj = tmp_path / "work"
        proj.mkdir()
        (proj / "rgpycrumbs.toml").write_text(
            "[dispatch]\nforce_uv = true\n"
            "[pins.packages]\njax = \"0.4.31\"\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.chdir(proj)
        monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)

        cfg = load_config(cwd=proj)
        assert cfg.force_uv is True  # project wins
        assert cfg.auto_deps is False  # from global
        assert cfg.package_pins["jax"] == "0.4.31"  # project overrides
        assert cfg.package_pins["numpy"] == "1.0.0"  # kept from global
        # lock path resolved relative to global config dir
        assert cfg.lock_path is not None
        assert cfg.lock_path.name == "global.lock"
        assert user in cfg.sources

    def test_explicit_config_env(self, monkeypatch, tmp_path):
        extra = tmp_path / "extra.toml"
        extra.write_text(
            "[pins.packages]\nscipy = \"1.14.1\"\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(CONFIG_PATH_ENV, str(extra))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty_xdg"))
        # no project
        monkeypatch.chdir(tmp_path)
        cfg = load_config(cwd=tmp_path)
        assert cfg.package_pins["scipy"] == "1.14.1"

    def test_missing_explicit_config_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CONFIG_PATH_ENV, str(tmp_path / "nope.toml"))
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_config(cwd=tmp_path)


class TestResolveLayers:
    def test_cli_beats_env_beats_config(self, monkeypatch, tmp_path):
        xdg = tmp_path / "xdg"
        (xdg / "rgpycrumbs").mkdir(parents=True)
        (xdg / "rgpycrumbs" / "config.toml").write_text(
            '[pins]\nlock = "from-config.lock"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.setenv(LOCK_PATH_ENV, str(tmp_path / "from-env.lock"))
        monkeypatch.chdir(tmp_path)

        cfg = load_config(cwd=tmp_path)
        # env over config
        p = resolve_lock_path_layered(config=cfg, cwd=tmp_path)
        assert p is not None
        assert p.name == "from-env.lock"
        # CLI over env
        p2 = resolve_lock_path_layered(
            cli_lock=str(tmp_path / "from-cli.lock"),
            config=cfg,
            cwd=tmp_path,
        )
        assert p2 is not None
        assert p2.name == "from-cli.lock"

    def test_force_uv_and_auto_deps(self, monkeypatch, tmp_path):
        xdg = tmp_path / "xdg"
        (xdg / "rgpycrumbs").mkdir(parents=True)
        (xdg / "rgpycrumbs" / "config.toml").write_text(
            "[dispatch]\nforce_uv = true\nauto_deps = false\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.delenv("RGPYCRUMBS_FORCE_UV", raising=False)
        monkeypatch.delenv("RGPYCRUMBS_AUTO_DEPS", raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_config(cwd=tmp_path)
        assert resolve_force_uv(is_dev=False, config=cfg) is True
        assert resolve_force_uv(is_dev=True, config=cfg) is False
        assert resolve_auto_deps_default(config=cfg) == "0"
        monkeypatch.setenv("RGPYCRUMBS_AUTO_DEPS", "1")
        assert resolve_auto_deps_default(config=cfg) == "1"


class TestDispatchUsesConfig:
    @patch("rgpycrumbs.cli.subprocess.run")
    def test_toml_package_pins_without_lock_file(self, mock_run, monkeypatch, tmp_path):
        from rgpycrumbs.cli import _dispatch
        from rgpycrumbs.locks import PINS_ENV
        import json

        xdg = tmp_path / "xdg"
        (xdg / "rgpycrumbs").mkdir(parents=True)
        (xdg / "rgpycrumbs" / "config.toml").write_text(
            "[pins.packages]\njax = \"0.4.31\"\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.delenv(LOCK_PATH_ENV, raising=False)
        monkeypatch.delenv(SBOM_PATH_ENV, raising=False)
        monkeypatch.setenv("RGPYCRUMBS_FORCE_UV", "1")
        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        monkeypatch.setattr("rgpycrumbs.cli.shutil.which", lambda name: "/usr/bin/uv")
        monkeypatch.chdir(tmp_path)

        _dispatch("group", "script", ())

        env = mock_run.call_args.kwargs["env"]
        pins = json.loads(env[PINS_ENV])
        assert pins[normalize_pypi_name("jax")] == "0.4.31"
        cmd = mock_run.call_args[0][0]
        assert "--constraints" in cmd
