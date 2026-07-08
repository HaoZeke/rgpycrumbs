# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Tests for optional CycloneDX SBOM consumption."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rgpycrumbs.sbom import (
    SBOM_PINS_ENV,
    apply_pin_to_spec,
    load_cyclonedx,
    load_pypi_pins,
    normalize_pypi_name,
    package_name_from_spec,
    pins_from_env,
    pins_to_constraint_lines,
    pypi_pins_from_cyclonedx,
)

pytestmark = pytest.mark.pure

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_pypi.cdx.json"


class TestCycloneDXParse:
    def test_fixture_exists_and_is_cyclonedx(self):
        doc = load_cyclonedx(FIXTURE)
        assert doc["bomFormat"] == "CycloneDX"
        assert isinstance(doc["components"], list)

    def test_pypi_pins_extracted_non_empty(self):
        pins = load_pypi_pins(FIXTURE)
        assert pins
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"
        assert pins[normalize_pypi_name("jax")] == "0.4.31"
        # purl uses adjusttext; normalize lookup
        assert pins[normalize_pypi_name("adjustText")] == "1.2.0"

    def test_skips_generic_and_versionless(self):
        pins = load_pypi_pins(FIXTURE)
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
        """eb-stack planned SBOMs use pkg:generic — skip, do not invent pip pins."""
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


class TestPinApply:
    def test_constraint_lines_sorted(self):
        lines = pins_to_constraint_lines({"jax": "0.4.31", "numpy": "2.0.2"})
        assert lines == ["jax==0.4.31", "numpy==2.0.2"]

    def test_apply_pin_to_spec(self):
        pins = {"jax": "0.4.31", "scipy": "1.14.0"}
        assert apply_pin_to_spec("jax>=0.4", pins) == "jax==0.4.31"
        assert package_name_from_spec("scipy>=1.11") == "scipy"
        assert package_name_from_spec("chemparseplot[neb,plot]>=1.8.0,<2") == (
            "chemparseplot"
        )
        assert apply_pin_to_spec("scipy>=1.11", pins) == "scipy==1.14.0"
        assert apply_pin_to_spec("ase>=3.22", pins) == "ase>=3.22"

    def test_pins_from_env_json(self, monkeypatch):
        monkeypatch.setenv(SBOM_PINS_ENV, json.dumps({"NumPy": "2.0.2"}))
        pins = pins_from_env()
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"

    def test_resolve_pip_spec_honors_sbom_pins(self, monkeypatch):
        """Shipped ensure_import path uses SBOM pins when env is set."""
        from rgpycrumbs._aux import _resolve_pip_spec

        monkeypatch.setenv(SBOM_PINS_ENV, json.dumps({"jax": "0.4.31"}))
        # force CUDA off path consistency
        monkeypatch.setattr("rgpycrumbs._aux._has_cuda", lambda: False)
        spec = _resolve_pip_spec("jax")
        assert spec == "jax==0.4.31"


class TestDispatchSbom:
    @patch("rgpycrumbs.cli.subprocess.run")
    def test_missing_sbom_exits(self, mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        with pytest.raises(SystemExit) as ei:
            _dispatch("group", "script", (), sbom_path="/no/such/sbom.cdx.json")
        assert ei.value.code == 1
        mock_run.assert_not_called()

    @patch("rgpycrumbs.cli.subprocess.run")
    def test_sbom_adds_uv_constraints_and_env_pins(self, mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch
        from rgpycrumbs.sbom import SBOM_PINS_ENV

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        monkeypatch.setenv("RGPYCRUMBS_FORCE_UV", "1")
        monkeypatch.setattr("rgpycrumbs.cli.shutil.which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr("rgpycrumbs.cli._in_env_stack_ready", lambda: False)

        _dispatch("group", "script", (), sbom_path=str(FIXTURE))

        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["uv", "run"]
        assert "--constraints" in cmd
        cidx = cmd.index("--constraints")
        cpath = Path(cmd[cidx + 1])
        # file is deleted in finally after run; mock run is sync so file existed at call
        # re-check pins via env passed to subprocess
        env = mock_run.call_args.kwargs["env"]
        pins = json.loads(env[SBOM_PINS_ENV])
        assert pins[normalize_pypi_name("numpy")] == "2.0.2"
        assert pins[normalize_pypi_name("jax")] == "0.4.31"

    @patch("rgpycrumbs.cli.subprocess.run")
    def test_no_sbom_unchanged_uv_command(self, mock_run, monkeypatch):
        from rgpycrumbs.cli import _dispatch
        from rgpycrumbs.sbom import SBOM_PINS_ENV, SBOM_PATH_ENV

        monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
        monkeypatch.delenv(SBOM_PATH_ENV, raising=False)
        monkeypatch.setenv("RGPYCRUMBS_FORCE_UV", "1")
        monkeypatch.setattr("rgpycrumbs.cli.shutil.which", lambda name: "/usr/bin/uv")

        _dispatch("group", "script", ())

        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["uv", "run"]
        assert "--constraints" not in cmd
        env = mock_run.call_args.kwargs["env"]
        assert SBOM_PINS_ENV not in env or not env.get(SBOM_PINS_ENV)
