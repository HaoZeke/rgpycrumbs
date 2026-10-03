# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Dispatcher routes and Click surfaces of the UMA AOTI scripts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from rgpycrumbs.uma import ExportEnv, aoti_key, package_path, sidecar_path
from rgpycrumbs.uma._cache import install

pytestmark = [
    pytest.mark.pure,
    pytest.mark.filterwarnings("ignore:.*is a dispatched PEP 723 script.*:UserWarning"),
]

ROOT = Path(__file__).resolve().parent.parent
ENV = ExportEnv(device="cpu", target="x86_64|AVX512|", torch="2.13.0", fairchem="2.23.0")


def test_prepare_aoti_declares_the_export_stack():
    text = (ROOT / "rgpycrumbs" / "uma" / "prepare_aoti.py").read_text()
    assert '# requires-python = ">=3.11"' in text
    assert "fairchem-core>=2.22" in text
    assert "vesin>=0.6" in text
    assert "readcon>=0.14.5" in text


def test_core_dependencies_stay_light():
    text = (ROOT / "pyproject.toml").read_text()
    core = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "torch" not in core
    assert "fairchem" not in core


def test_aoti_cache_needs_only_click():
    text = (ROOT / "rgpycrumbs" / "uma" / "aoti_cache.py").read_text()
    header = text.split("# /// script", 1)[1].split("# ///", 1)[0]
    deps = [line for line in header.splitlines() if line.strip().startswith('#   "')]
    assert deps == ['#   "click>=8.1",']


@pytest.mark.parametrize(
    ("command", "script"),
    [("prepare-aoti", "uma/prepare_aoti.py"), ("aoti-cache", "uma/aoti_cache.py")],
)
@patch("rgpycrumbs.cli.subprocess.run")
def test_help_routes_through_dispatcher(mock_run, monkeypatch, command, script):
    from rgpycrumbs.cli import main

    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setattr("rgpycrumbs.cli._uv_editable_sources", lambda: [])

    result = CliRunner().invoke(main, ["uma", command, "--help"])
    assert result.exit_code == 0, result.exception or result.output

    run = mock_run.call_args.args[0]
    assert any(str(part).endswith(script) for part in run)
    assert run[-1] == "--help"


def test_uma_group_lists_only_the_scripts():
    from rgpycrumbs.cli import main

    result = CliRunner().invoke(main, ["uma", "--help"])
    assert result.exit_code == 0
    assert "prepare-aoti" in result.output
    assert "aoti-cache" in result.output
    for private in ("probe", "cache ", "key", "prepare "):
        assert f"  {private}" not in result.output


def _populate(cache: Path, *, spin_in_package: int = 1) -> Path:
    key = aoti_key([6, 7, 1], ENV)
    work = cache / "work"
    work.mkdir(parents=True)
    built = work / f"{key.stem()}.pt2"
    import zipfile

    meta = {
        "task_name": "omol",
        "charge": "0",
        "spin": str(spin_in_package),
        "z_set": "[1, 6, 7]",
        "shapes": "{'pos': [3, 3]}",
        "pos_dtype": "float32",
        "molecular_box": "0.0",
        "batch_max": "0",
        "AOTI_DEVICE_KEY": "cpu",
        "AOTI_MACHINE": "x86_64",
        "AOTI_CPU_ISA": "AVX512",
        "AOTI_COMPUTE_CAPABILITY": "",
    }
    with zipfile.ZipFile(built, "w") as zf:
        zf.writestr(
            f"{key.stem()}/data/aotinductor/model/a.wrapper_metadata.json",
            json.dumps(meta),
        )
    return install(cache, key, built, {"compile_seconds": 12.5, "embedded": meta})


class TestAotiCache:
    def _run(self, *args):
        from rgpycrumbs.uma.aoti_cache import main

        return CliRunner().invoke(main, list(args))

    def test_list(self, tmp_path):
        pt2 = _populate(tmp_path)
        out = self._run("list", "--cache", str(tmp_path))
        assert out.exit_code == 0, out.output
        assert "1 package(s)" in out.output
        assert "CHN q=0 s=1 task=omol" in out.output
        assert "compiled 12.5 s" in out.output
        rows = self._run("list", "--cache", str(tmp_path), "--json").output.splitlines()
        assert json.loads(rows[0])["path"] == str(pt2)

    def test_inspect_ok_and_mismatch(self, tmp_path):
        good = tmp_path / "good"
        bad = tmp_path / "bad"
        pt2 = _populate(good)
        out = self._run("inspect", "--cache", str(good), "omol-CHN")
        assert out.exit_code == 0, out.output
        assert "matches the key" in out.output
        _populate(bad, spin_in_package=3)
        out = self._run("inspect", "--cache", str(bad), "omol-CHN")
        assert out.exit_code == 1
        assert "MISMATCH spin" in out.output
        assert self._run("inspect", str(pt2)).exit_code == 0

    def test_inspect_unknown_name(self, tmp_path):
        out = self._run("inspect", "--cache", str(tmp_path), "nothing")
        assert out.exit_code == 1
        assert "no cache entry" in out.output

    def test_clear(self, tmp_path):
        pt2 = _populate(tmp_path)
        assert self._run("clear", "--cache", str(tmp_path)).exit_code == 2
        out = self._run("clear", "--cache", str(tmp_path), "--all", "--dry-run")
        assert "would remove" in out.output
        assert pt2.exists()
        out = self._run("clear", "--cache", str(tmp_path), "--all", "--partial")
        assert out.exit_code == 0, out.output
        assert not pt2.exists()
        assert not sidecar_path(pt2).exists()
        assert "removed work" not in out.output


class TestPrepareAotiCli:
    @pytest.fixture
    def cli(self, monkeypatch):
        import types

        from rgpycrumbs.uma import prepare_aoti

        atoms = types.SimpleNamespace(info={}, get_atomic_numbers=lambda: [6, 7, 1])
        monkeypatch.setattr(prepare_aoti, "load_atoms", lambda path: atoms)
        monkeypatch.setattr("rgpycrumbs.uma._prepare.probe_env", lambda *a, **k: ENV)
        return prepare_aoti.main

    def test_hit_prints_the_path_last(self, tmp_path, cli):
        pt2 = _populate(tmp_path)
        structure = tmp_path / "hcn.xyz"
        structure.write_text("3\n\n")
        out = CliRunner().invoke(cli, [str(structure), "--cache", str(tmp_path)])
        assert out.exit_code == 0, out.output
        assert out.output.splitlines()[-1] == str(pt2)
        assert "hit" in out.output

    def test_dry_run_reports_the_miss(self, tmp_path, cli):
        exporter = tmp_path / "export_uma_aoti.py"
        exporter.write_text("# exporter\n")
        structure = tmp_path / "hcn.xyz"
        structure.write_text("3\n\n")
        out = CliRunner().invoke(
            cli,
            [
                str(structure),
                "--cache",
                str(tmp_path),
                "--exporter",
                str(exporter),
                "--molecular-box",
                "25",
                "--dry-run",
            ],
        )
        assert out.exit_code == 0, out.output
        assert "miss" in out.output
        assert "would run:" in out.output
        assert "--molecular-box 25" in out.output
        key = aoti_key([6, 7, 1], ENV, molecular_box=25)
        assert out.output.splitlines()[-1] == str(package_path(tmp_path, key))
        assert not package_path(tmp_path, key).exists()

    def test_parity_error_is_a_click_error(self, tmp_path, cli):
        structure = tmp_path / "hcn.xyz"
        structure.write_text("3\n\n")
        out = CliRunner().invoke(
            cli, [str(structure), "--cache", str(tmp_path), "--spin", "2"]
        )
        assert out.exit_code == 1
        assert "Error: spin multiplicity 2 is impossible" in out.output
