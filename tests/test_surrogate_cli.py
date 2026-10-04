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


def test_scaling_csv_writes_a_figure_per_cell_and_pop_panels(tmp_path):
    wall = tmp_path / "w.csv"
    rows = ["cell,set,ranks,threads,repetition,stage,seconds,calls,partition"]
    for cell in ("a", "b"):
        for ranks, thr, v, calls in ((1, 1, 10, 50), (2, 1, 6, 50), (1, 2, 7, 60)):
            rows.append(f"{cell},s,{ranks},{thr},1,pipeline,{v},,p")
            rows.append(f"{cell},s,{ranks},{thr},1,search,{v - 1},{calls},p")
    wall.write_text("\n".join(rows) + "\n")
    pop = tmp_path / "p.csv"
    pop.write_text("cell,ranks,threads,lb\na,1,1,1\na,2,1,0.8\nb,1,1,1\nb,2,1,0.7\n")
    result = CliRunner().invoke(
        _load("plt-scaling"),
        [str(wall), "-o", str(tmp_path / "s"), "--pop", str(pop), "--pop-metric", "lb"],
    )
    assert result.exit_code == 0, result.output
    for name in ("speedup", "efficiency", "a-strong", "b-strong", "a-pop", "b-pop"):
        assert (tmp_path / f"s-{name}.png").is_file(), name


def test_scaling_pop_needs_a_metric(tmp_path):
    t = tmp_path / "t.json"
    t.write_text('{"x": {"workers": [1, 2], "time_s": [2, 1]}}')
    p = tmp_path / "p.csv"
    p.write_text("ranks,threads,lb\n1,1,1\n")
    result = CliRunner().invoke(
        _load("plt-scaling"), [str(t), "-o", str(tmp_path / "s"), "--pop", str(p)]
    )
    assert result.exit_code != 0 and "--pop-metric" in result.output


def test_font_option_embeds_the_family_and_names_missing_ones(tmp_path):
    import re
    import shutil

    import matplotlib.font_manager as fm

    fonts = tmp_path / "fonts"
    fonts.mkdir()
    shutil.copy(fm.findfont("DejaVu Serif"), fonts / "DejaVuSerif.ttf")
    table = tmp_path / "t.json"
    table.write_text('{"x": {"workers": [1, 2], "time_s": [2, 1]}}')
    main = _load("plt-scaling")
    args = [str(table), "-o", str(tmp_path / "s"), "--format", "pdf"]
    ok = CliRunner().invoke(
        main, [*args, "--font", "DejaVu Serif", "--font-dir", str(fonts)]
    )
    assert ok.exit_code == 0, ok.output
    raw = (tmp_path / "s-speedup.pdf").read_bytes()
    names = {
        n.decode()
        for n in re.findall(rb"/FontName\s*/(?:[A-Z]{6}\+)?([^\s/>\[\]]+)", raw)
    }
    assert names and all(n.startswith("DejaVuSerif") for n in names), names
    bad = CliRunner().invoke(
        main, [*args, "--font", "No Such 123", "--font-dir", str(fonts)]
    )
    assert (
        bad.exit_code != 0
        and "No Such 123" in str(bad.exception)
        and str(fonts) in str(bad.exception)
    )
