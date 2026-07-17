# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Tests for lock / SBOM consumption (uv.lock, pylock, CycloneDX)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rgpycrumbs.locks import (
    LOCK_PATH_ENV,
    PINS_ENV,
    SBOM_PATH_ENV,
    SBOM_PINS_ENV,
    LockFormat,
    apply_pin_to_spec,
    detect_lock_format,
    load_cyclonedx,
    load_pypi_pins,
    normalize_pypi_name,
    package_name_from_spec,
    pins_from_env,
    pins_to_constraint_lines,
    pypi_pins_from_cyclonedx,
    pypi_pins_from_pylock,
    pypi_pins_from_uv_lock,
)

pytestmark = pytest.mark.pure

FIXTURES = Path(__file__).parent / "fixtures"
CDX = FIXTURES / "minimal_pypi.cdx.json"
UV_LOCK = FIXTURES / "minimal_uv.lock"
PYLOCK = FIXTURES / "minimal_pylock.toml"


class TestCycloneDXParse:
    def test_fixture_exists_and_is_cyclonedx(self):
        doc = load_cyclonedx(CDX)
        assert doc["bomFormat"] == "CycloneDX"
        assert isinstance(doc["components"], list)

    def test_pypi_pins_extracted_non_empty(self):
        pins = load_pypi_pins(CDX)
        assert pins
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"
        assert pins[normalize_pypi_name("jax")] == "0.4.31"
        assert pins[normalize_pypi_name("adjustText")] == "1.2.0"

    def test_skips_generic_and_versionless(self):
        pins = load_pypi_pins(CDX)
        assert "gromacs" not in pins
        assert "noversion" not in pins

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_pypi_pins(tmp_path / "nope.cdx.json")

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_cyclonedx(bad)

    def test_wrong_bom_format_raises(self, tmp_path):
        bad = tmp_path / "spdx.json"
        bad.write_text(json.dumps({"bomFormat": "SPDX", "components": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported bomFormat"):
            load_cyclonedx(bad)

    def test_eb_stack_generic_only_yields_empty_pins(self):
        doc = {
            "bomFormat": "CycloneDX",
            "components": [
                {
                    "name": "GROMACS",
                    "version": "2025.0",
                    "purl": "pkg:generic/GROMACS@2025.0?toolchain=foss-2025b",
                }
            ],
        }
        assert pypi_pins_from_cyclonedx(doc) == {}


class TestPylockAndUvLock:
    def test_detect_formats(self):
        assert detect_lock_format(UV_LOCK) is LockFormat.UV_LOCK
        assert detect_lock_format(PYLOCK) is LockFormat.PYLOCK
        assert detect_lock_format(CDX) is LockFormat.CYCLONEDX

    def test_uv_lock_pins(self):
        pins = load_pypi_pins(UV_LOCK)
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"
        assert pins[normalize_pypi_name("jax")] == "0.4.31"
        assert pins[normalize_pypi_name("click")] == "8.1.8"

    def test_pylock_pins(self):
        pins = load_pypi_pins(PYLOCK)
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"
        assert pins[normalize_pypi_name("scipy")] == "1.14.1"

    def test_pylock_named_file(self, tmp_path):
        dest = tmp_path / "pylock.dev.toml"
        dest.write_text(PYLOCK.read_text(encoding="utf-8"), encoding="utf-8")
        assert detect_lock_format(dest) is LockFormat.PYLOCK
        pins = load_pypi_pins(dest)
        assert "numpy" in pins

    def test_direct_parsers(self):
        import tomllib

        uv_doc = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
        assert "numpy" in pypi_pins_from_uv_lock(uv_doc)
        py_doc = tomllib.loads(PYLOCK.read_text(encoding="utf-8"))
        assert "scipy" in pypi_pins_from_pylock(py_doc)


class TestPinApply:
    def test_constraint_lines_sorted(self):
        lines = pins_to_constraint_lines({"jax": "0.4.31", "numpy": "2.0.2"})
        assert lines == ["jax==0.4.31", "numpy==2.0.2"]

    def test_apply_pin_to_spec(self):
        pins = {"jax": "0.4.31", "scipy": "1.14.0"}
        assert apply_pin_to_spec("jax>=0.4", pins) == "jax==0.4.31"
        assert package_name_from_spec("scipy>=1.11") == "scipy"
        assert package_name_from_spec("chemparseplot[neb,plot]>=1.9.13,<2") == (
            "chemparseplot"
        )
        assert apply_pin_to_spec("scipy>=1.11", pins) == "scipy==1.14.0"
        assert apply_pin_to_spec("ase>=3.22", pins) == "ase>=3.22"

    def test_pins_from_env_json(self, monkeypatch):
        monkeypatch.setenv(PINS_ENV, json.dumps({"NumPy": "2.0.2"}))
        pins = pins_from_env()
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"

    def test_pins_from_legacy_sbom_env(self, monkeypatch):
        monkeypatch.delenv(PINS_ENV, raising=False)
        monkeypatch.setenv(SBOM_PINS_ENV, json.dumps({"jax": "0.4.31"}))
        assert pins_from_env()[normalize_pypi_name("jax")] == "0.4.31"

    def test_resolve_pip_spec_honors_lock_pins(self, monkeypatch):
        from rgpycrumbs._aux import _resolve_pip_spec

        monkeypatch.setenv(PINS_ENV, json.dumps({"jax": "0.4.31"}))
        monkeypatch.setattr("rgpycrumbs._aux._has_cuda", lambda: False)
        assert _resolve_pip_spec("jax") == "jax==0.4.31"


class TestDispatchLock:
    @patch("rgpycrumbs.cli.subprocess.run")
    def test_missing_lock_exits(self, mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        with pytest.raises(SystemExit) as ei:
            _dispatch("group", "script", (), lock_path="/no/such/uv.lock")
        assert ei.value.code == 1
        mock_run.assert_not_called()

    @patch("rgpycrumbs.cli.subprocess.run")
    def test_uv_lock_adds_constraints_and_env_pins(self, mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        monkeypatch.setenv("RGPYCRUMBS_FORCE_UV", "1")
        monkeypatch.setattr("rgpycrumbs.cli.shutil.which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr("rgpycrumbs.cli._in_env_stack_ready", lambda: False)

        _dispatch("group", "script", (), lock_path=str(UV_LOCK))

        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["uv", "run"]
        assert "--constraints" in cmd
        env = mock_run.call_args.kwargs["env"]
        pins = json.loads(env[PINS_ENV])
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"
        assert env.get("UV_CONSTRAINT")

    @patch("rgpycrumbs.cli.subprocess.run")
    def test_pylock_and_sbom_alias(self, mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        monkeypatch.setenv("RGPYCRUMBS_FORCE_UV", "1")
        monkeypatch.setattr("rgpycrumbs.cli.shutil.which", lambda name: "/usr/bin/uv")

        _dispatch("group", "script", (), sbom_path=str(CDX))
        env = mock_run.call_args.kwargs["env"]
        pins = json.loads(env[SBOM_PINS_ENV])
        assert pins[normalize_pypi_name("jax")] == "0.4.31"

        mock_run.reset_mock()
        _dispatch("group", "script", (), lock_path=str(PYLOCK))
        env = mock_run.call_args.kwargs["env"]
        pins = json.loads(env[PINS_ENV])
        assert pins[normalize_pypi_name("scipy")] == "1.14.1"

    @patch("rgpycrumbs.cli.subprocess.run")
    def test_no_lock_unchanged_uv_command(self, mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        monkeypatch.delenv(LOCK_PATH_ENV, raising=False)
        monkeypatch.delenv(SBOM_PATH_ENV, raising=False)
        monkeypatch.setenv("RGPYCRUMBS_FORCE_UV", "1")
        monkeypatch.setattr("rgpycrumbs.cli.shutil.which", lambda name: "/usr/bin/uv")

        _dispatch("group", "script", ())

        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["uv", "run"]
        assert "--constraints" not in cmd
        env = mock_run.call_args.kwargs["env"]
        assert not env.get(PINS_ENV)
