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
COMMANDS = [
    "plt-band",
    "plt-history",
    "plt-dimer",
    "plt-campaign",
    "plt-scaling",
    "plt-breakdown",
    "plt-cases",
]


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


def test_breakdown_writes_one_figure_per_cell(tmp_path):
    rows = ["cell,set,ranks,threads,repetition,stage,seconds,calls,partition"]
    for cell in ("a", "b"):
        for ranks, tot in ((1, 10), (2, 6)):
            for stage, v in (
                ("pipeline", tot),
                ("band_oracle", tot / 2),
                ("band_refit", tot / 4),
            ):
                rows.append(f"{cell},s,{ranks},1,1,{stage},{v},,p")
    wall = tmp_path / "w.csv"
    wall.write_text("\n".join(rows) + "\n")
    result = CliRunner().invoke(
        _load("plt-breakdown"),
        [
            str(wall),
            "-o",
            str(tmp_path / "b"),
            "--component",
            "band_oracle",
            "--component",
            "band_refit",
        ],
    )
    assert result.exit_code == 0, result.output
    for cell in ("a", "b"):
        assert (tmp_path / f"b-{cell}-breakdown.png").is_file()
    bad = CliRunner().invoke(
        _load("plt-breakdown"),
        [str(wall), "-o", str(tmp_path / "c"), "--component", "nope"],
    )
    assert bad.exit_code != 0


def test_cases_writes_calls_and_wall_and_rejects_bad_baselines(tmp_path):
    cases = tmp_path / "c.csv"
    cases.write_text(
        "board,case,label,search_calls,validation_calls,pipeline_wall_s,certified,reference_pipeline_wall_s\n"
        "B1,a,Alpha,30,20,6.5,true,8\nB1,b,Beta,50,25,12,true,\nB2,c,Gamma,90,30,300,true,\n"
    )
    base = tmp_path / "b.csv"
    base.write_text("board,case,method,calls,converged\nB1,a,M,100,false\n")
    main = _load("plt-cases")
    ok = CliRunner().invoke(
        main,
        [
            str(cases),
            "-o",
            str(tmp_path / "x"),
            "--baseline",
            str(base),
            "--reference-label",
            "earlier record",
            "--format",
            "svg",
        ],
    )
    assert ok.exit_code == 0, ok.output
    for name in ("calls", "wall"):
        assert (tmp_path / f"x-{name}.svg").is_file()
    base.write_text("board,case,method,calls,converged\nB1,zzz,M,1,true\n")
    bad = CliRunner().invoke(
        main, [str(cases), "-o", str(tmp_path / "y"), "--baseline", str(base)]
    )
    assert bad.exit_code != 0 and "unknown case" in bad.output


def test_title_option_is_accepted_by_the_search_commands():
    for command in ("plt-band", "plt-history", "plt-dimer"):
        main = _load(command)
        assert "title" in {p.name for p in main.params}, command


def test_band_observation_options_exist():
    names = {p.name for p in _load("plt-band").params}
    assert {
        "profile_observations",
        "landscape_surface",
        "landscape_window",
        "landscape_xlim",
        "landscape_ylim",
        "label_critical_points",
        "plot_structures",
        "n_structures",
        "types_from",
        "strip_renderer",
        "landscape_color",
        "landscape_fade_variance",
        "landscape_label_every",
        "observation_mode",
        "observation_distance",
        "title",
    } <= names


def test_band_strip_needs_types_and_accepts_a_types_file(tmp_path):
    import shutil

    pytest.importorskip("ase")
    pytest.importorskip("h5py")
    pytest.importorskip("jax")
    fixture = (
        Path(__file__).resolve().parent.parent.parent
        / "chemparseplot/tests/fixtures/surrogate/record/baker/25_hcnh2"
    )
    if not fixture.is_dir():
        pytest.skip("chemparseplot fixture record not next to this checkout")
    cell = tmp_path / "baker" / "25_hcnh2"
    shutil.copytree(fixture, cell)
    (cell / "saddle" / "pos.con").unlink()
    main = _load("plt-band")
    args = [
        str(cell),
        "-o",
        str(tmp_path / "b"),
        "--panel",
        "profile",
        "--strip-renderer",
        "ase",
    ]
    bad = CliRunner().invoke(main, args)
    assert (
        bad.exit_code != 0
        and "no atom types" in bad.output
        and "--types-from" in bad.output
    )
    xyz = tmp_path / "t.xyz"
    xyz.write_text("5\n\nC 0 0 0\nN 1 0 0\nH 0 1 0\nH 0 0 1\nH 1 1 0\n")
    ok = CliRunner().invoke(main, [*args, "--types-from", str(xyz)])
    assert ok.exit_code == 0, ok.output
    assert (tmp_path / "b-profile.png").is_file()


@pytest.mark.parametrize(
    "command",
    [c for c in COMMANDS if c != "plt-band"] + ["plt-band"],
)
def test_every_figure_command_takes_legend_fontsize(command):
    assert "legend_fontsize" in {p.name for p in _load(command).params}


def test_legend_fontsize_and_critical_labels_reach_the_figure(tmp_path):
    import shutil

    pytest.importorskip("ase")
    pytest.importorskip("h5py")
    pytest.importorskip("jax")
    fixture = (
        Path(__file__).resolve().parent.parent.parent
        / "chemparseplot/tests/fixtures/surrogate/record/baker/25_hcnh2"
    )
    if not fixture.is_dir():
        pytest.skip("chemparseplot fixture record not next to this checkout")
    cell = tmp_path / "baker" / "25_hcnh2"
    shutil.copytree(fixture, cell)
    main = _load("plt-band")
    args = [
        str(cell),
        "-o",
        str(tmp_path / "b"),
        "--panel",
        "profile",
        "--plot-structures",
        "none",
        "--format",
        "svg",
    ]
    plain = CliRunner().invoke(main, args)
    assert plain.exit_code == 0, plain.output
    labelled = CliRunner().invoke(
        main,
        [
            *args[:2],
            str(tmp_path / "c"),
            *args[3:],
            "--label-critical-points",
            "--legend-fontsize",
            "10",
        ],
    )
    assert labelled.exit_code == 0, labelled.output
    a = (tmp_path / "b-profile.svg").read_text()
    b = (tmp_path / "c-profile.svg").read_text()
    assert a != b  # the letters and the larger legend change the drawing
