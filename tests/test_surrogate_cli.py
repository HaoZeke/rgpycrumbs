# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Dispatcher routes and options of the surrogate-search figure commands."""

import importlib.util
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = ["plt-band", "plt-history", "plt-dimer", "plt-campaign", "plt-scaling"]


def _script(command):
    return ROOT / "rgpycrumbs" / "surrogate" / f"{command.replace('-', '_')}.py"


@pytest.mark.parametrize("command", COMMANDS)
def test_inline_metadata_pins_chemparseplot_to_a_commit(command):
    text = _script(command).read_text()
    block = text.split("# /// script")[1].split("# ///")[0]
    assert 'requires-python = ">=3.11"' in block
    assert re.search(
        r"chemparseplot @ git\+https://github.com/HaoZeke/chemparseplot@[0-9a-f]{40}",
        block,
    )
    assert "sys.path" not in text


@pytest.mark.parametrize("command", COMMANDS)
@patch("rgpycrumbs.cli.subprocess.run")
def test_help_routes_through_dispatcher(mock_run, command, monkeypatch):
    from rgpycrumbs.cli import main

    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setattr("rgpycrumbs.cli._uv_editable_sources", lambda: [])

    result = CliRunner().invoke(main, ["surrogate", command, "--help"])
    assert result.exit_code == 0, result.exception or result.output
    cmd = mock_run.call_args.args[0]
    assert any(str(p).endswith(f"surrogate/{command.replace('-', '_')}.py") for p in cmd)
    assert cmd[-1] == "--help"


def _load(command):
    pytest.importorskip("chemparseplot.plot.surrogate")
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location(f"_t_{command}", _script(command))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def test_scaling_writes_both_figures_with_provenance(tmp_path):
    table = tmp_path / "t.json"
    table.write_text(
        json.dumps(
            {
                "ours": {"workers": [1, 2, 4], "time_s": [10.0, 6.0, 3.5]},
                "other": {"workers": [1, 2], "time_s": [10.0, 9.0]},
            }
        )
    )
    result = CliRunner().invoke(
        _load("plt-scaling"), [str(table), "-o", str(tmp_path / "s"), "--format", "svg"]
    )
    assert result.exit_code == 0, result.output
    for name in ("speedup", "efficiency"):
        f = tmp_path / f"s-{name}.svg"
        assert f.is_file()
        side = json.loads((tmp_path / f"s-{name}.svg.provenance.json").read_text())
        assert side["inputs"]["t.json"] and side["output"]["sha256"]


def test_scaling_rejects_a_malformed_table(tmp_path):
    table = tmp_path / "t.json"
    table.write_text('{"ours": [1, 2]}')
    result = CliRunner().invoke(
        _load("plt-scaling"), [str(table), "-o", str(tmp_path / "s")]
    )
    assert result.exit_code != 0 and "workers" in result.output


def test_campaign_baseline_option_must_name_a_file(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    result = CliRunner().invoke(
        _load("plt-campaign"),
        [str(rec), "-o", str(tmp_path / "c"), "--baseline", "x=missing.json"],
    )
    assert result.exit_code != 0 and "NAME=FILE" in result.output


def test_scaling_reads_per_repetition_csv(tmp_path):
    wall = tmp_path / "w.csv"
    rows = ["cell,set,ranks,threads,repetition,stage,seconds,calls,partition"]
    for ranks, secs in ((1, (10, 11)), (2, (6, 7))):
        rows += [f"c,s,{ranks},1,{i + 1},pipeline,{v},50,p" for i, v in enumerate(secs)]
    wall.write_text("\n".join(rows) + "\n")
    result = CliRunner().invoke(
        _load("plt-scaling"), [str(wall), "-o", str(tmp_path / "s")]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "s-speedup.png").is_file()
