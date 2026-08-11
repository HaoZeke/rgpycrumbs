# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Cover plt-con without importing matplotlib at collection time."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.pure


class _FakeTable:
    def __init__(self, columns, height, null_count=0, empty=False):
        self.columns = columns
        self.height = height
        self._null_count = null_count
        self._empty = empty

    def is_empty(self):
        return self._empty

    def __getitem__(self, key):
        return types.SimpleNamespace(null_count=lambda: self._null_count)


class _FakeTraj:
    def __init__(self, *, energies, n_frames, source, table):
        self.energies = energies
        self.n_frames = n_frames
        self.source = source
        self.table = table


def _install_plot_stubs(monkeypatch):
    fig = MagicMock(name="fig")
    ax = MagicMock(name="ax")

    mpl = types.ModuleType("matplotlib")
    plt = types.ModuleType("matplotlib.pyplot")
    plt.subplots = MagicMock(return_value=(fig, ax))
    plt.close = MagicMock()
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)

    cpp = types.ModuleType("chemparseplot")
    cpp_api = types.ModuleType("chemparseplot.api")
    cpp_api.load_trajectory = MagicMock()
    cpp_api.plot_con_overview = MagicMock(return_value=fig)
    cpp_api.plot_con_overlay = MagicMock()
    cpp.api = cpp_api
    monkeypatch.setitem(sys.modules, "chemparseplot", cpp)
    monkeypatch.setitem(sys.modules, "chemparseplot.api", cpp_api)

    rich = types.ModuleType("rich")
    rich_logging = types.ModuleType("rich.logging")

    class _Handler:
        def __init__(self, *a, **k):
            pass

    rich_logging.RichHandler = _Handler
    rich.logging = rich_logging
    monkeypatch.setitem(sys.modules, "rich", rich)
    monkeypatch.setitem(sys.modules, "rich.logging", rich_logging)

    return fig, ax, cpp_api, plt


def _load_plt_con(monkeypatch):
    _install_plot_stubs(monkeypatch)
    sys.modules.pop("rgpycrumbs.eon.plt_con", None)
    return importlib.import_module("rgpycrumbs.eon.plt_con")


def test_has_forces_true_and_false(monkeypatch):
    plt_con = _load_plt_con(monkeypatch)
    good = _FakeTraj(
        energies=[0.0],
        n_frames=1,
        source="con",
        table=_FakeTable(["fmax"], height=2, null_count=0),
    )
    assert plt_con._has_forces(good) is True
    missing = _FakeTraj(
        energies=[0.0],
        n_frames=1,
        source="con",
        table=_FakeTable(["energy"], height=2),
    )
    assert plt_con._has_forces(missing) is False
    empty = _FakeTraj(
        energies=[0.0],
        n_frames=1,
        source="con",
        table=_FakeTable(["fmax"], height=0, empty=True),
    )
    assert plt_con._has_forces(empty) is False
    none_table = _FakeTraj(energies=[0.0], n_frames=1, source="con", table=None)
    assert plt_con._has_forces(none_table) is False
    all_null = _FakeTraj(
        energies=[0.0],
        n_frames=1,
        source="con",
        table=_FakeTable(["fmax"], height=2, null_count=2),
    )
    assert plt_con._has_forces(all_null) is False


def test_single_traj_overview(monkeypatch, tmp_path):
    fig, _ax, cpp_api, plt = _install_plot_stubs(monkeypatch)
    sys.modules.pop("rgpycrumbs.eon.plt_con", None)
    plt_con = importlib.import_module("rgpycrumbs.eon.plt_con")

    con = tmp_path / "movie.con"
    con.write_text("x\n")
    traj = _FakeTraj(
        energies=[1.0, float("nan")],
        n_frames=2,
        source="con",
        table=_FakeTable(["fmax"], height=2, null_count=0),
    )
    cpp_api.load_trajectory.return_value = traj

    from click.testing import CliRunner

    result = CliRunner().invoke(plt_con.main, [str(con)])
    assert result.exit_code == 0, result.output
    cpp_api.plot_con_overview.assert_called_once()
    kwargs = cpp_api.plot_con_overview.call_args.kwargs
    assert kwargs["relative"] is True
    assert kwargs["show_forces"] is True
    fig.savefig.assert_called_once()
    plt.close.assert_called_once_with(fig)
    out = tmp_path / "movie_profile.pdf"
    assert out == Path(fig.savefig.call_args.args[0])


def test_multi_file_overlay_and_label_mismatch(monkeypatch, tmp_path):
    _fig, _ax, cpp_api, plt = _install_plot_stubs(monkeypatch)
    sys.modules.pop("rgpycrumbs.eon.plt_con", None)
    plt_con = importlib.import_module("rgpycrumbs.eon.plt_con")

    a = tmp_path / "a.con"
    b = tmp_path / "b.con"
    a.write_text("a\n")
    b.write_text("b\n")
    traj = _FakeTraj(
        energies=[0.0],
        n_frames=1,
        source="con",
        table=_FakeTable(["energy"], height=1),
    )
    cpp_api.load_trajectory.return_value = traj

    from click.testing import CliRunner

    runner = CliRunner()
    bad = runner.invoke(plt_con.main, [str(a), str(b), "--label", "only-one"])
    assert bad.exit_code != 0
    assert "must match" in (bad.output + str(bad.exception))

    ok = runner.invoke(
        plt_con.main,
        [
            str(a),
            str(b),
            "--label",
            "A",
            "--label",
            "B",
            "--absolute",
            "-o",
            str(tmp_path / "ov.pdf"),
        ],
    )
    assert ok.exit_code == 0, ok.output
    cpp_api.plot_con_overlay.assert_called_once()
    kwargs = cpp_api.plot_con_overlay.call_args.kwargs
    assert kwargs["labels"] == ["A", "B"]
    assert kwargs["relative"] is False
    plt.subplots.assert_called_once()


def test_plt_con_script_declares_python_floor():
    script = Path(__file__).resolve().parent.parent / "rgpycrumbs" / "eon" / "plt_con.py"
    text = script.read_text()
    assert '# requires-python = ">=3.11"' in text


def test_plt_con_is_discovered_as_eon_script():
    from rgpycrumbs.cli import _get_scripts_in_folder

    assert "plt_con" in _get_scripts_in_folder("eon")
