"""Regression tests, one or more per fixed defect."""

from __future__ import annotations

import gzip
import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.pure

PROJECT_ROOT = Path(__file__).parents[1]
COG_BLOCK = re.compile(
    r"^\[\[\[cog\n(?P<code>.*?)^\]\]\]\n(?P<output>.*?)^\[\[\[end\]\]\]$",
    re.MULTILINE | re.DOTALL,
)


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


def test_pixi_test_tasks_propagate_test_failures() -> None:
    """Coverage orchestration must not turn failed test runs into success."""
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((PROJECT_ROOT / "pixi.toml").read_text())

    test_commands = {
        name: task["cmd"]
        for feature in project["feature"].values()
        for name, task in feature.get("tasks", {}).items()
        if name.startswith("test-")
    }

    assert set(test_commands) == {
        "test-base",
        "test-eonmlflow",
        "test-fragments",
        "test-ptm",
        "test-surfaces",
    }
    for command in test_commands.values():
        assert "; true" not in command


def test_ptm_environment_takes_ovito_from_its_channel() -> None:
    """conda-forge's ovito is the desktop application without a Python module."""
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((PROJECT_ROOT / "pixi.toml").read_text())
    ptm = project["feature"]["ptm"]

    assert ptm["channels"] == ["https://conda.ovito.org"]
    assert ptm["dependencies"]["ovito"]["channel"] == "https://conda.ovito.org"
    assert "ovito" not in project["feature"]["test"]["dependencies"]


def test_coverage_workflow_combines_every_cov_environment() -> None:
    """CI runs and combines each per-environment file that ``pixi run cov`` does."""
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((PROJECT_ROOT / "pixi.toml").read_text())
    cov = project["feature"]["test"]["tasks"]["cov"]["cmd"]
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci_coverage.yml").read_text()
    combine = next(line for line in workflow.splitlines() if "coverage combine" in line)

    environments = re.findall(r"\.coverage\.(\w+)", cov)
    assert environments
    for environment in environments:
        assert f"--data-file=.coverage.{environment} " in workflow
        assert f" .coverage.{environment}" in combine


def test_docs_export_regenerates_cog_blocks_first() -> None:
    """Org sources with cog blocks are regenerated before the Emacs export."""
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((PROJECT_ROOT / "pixi.toml").read_text())
    docs = project["feature"]["docs"]
    command = docs["tasks"]["mkrst"]["cmd"]

    assert "cogapp" in docs["pypi-dependencies"]
    assert command.index("cog -r") < command.index("emacs --batch")


def test_org_cog_output_matches_the_pixi_files() -> None:
    """Committed cog output in the org sources is what the blocks generate now."""
    pytest.importorskip("tomllib")
    blocks = 0
    for path in sorted((PROJECT_ROOT / "docs" / "orgmode").rglob("*.org")):
        for block in COG_BLOCK.finditer(path.read_text()):
            emitted: list[str] = []
            cog = SimpleNamespace(inFile=str(path), outl=emitted.append)
            exec(block["code"], {"cog": cog})  # noqa: S102
            assert block["output"] == "".join(f"{line}\n" for line in emitted), path
            blocks += 1
    assert blocks


def test_chemparseplot_requirements_share_one_released_floor() -> None:
    """Installs, pixi, docs and script headers take one floor or a commit."""
    tomllib = pytest.importorskip("tomllib")
    from rgpycrumbs._aux import _DEPENDENCY_MAP, _install_spec

    spec = _DEPENDENCY_MAP["chemparseplot"]
    floor = re.fullmatch(r"chemparseplot\[neb,plot\]>=([\d.]+),<2", spec)[1]
    pixi = tomllib.loads((PROJECT_ROOT / "pixi.toml").read_text())

    assert _install_spec(spec) == spec
    for feature in ("docs", "test"):
        requirement = pixi["feature"][feature]["pypi-dependencies"]["chemparseplot"]
        assert requirement == f">={floor}, <2", feature
    for path in sorted((PROJECT_ROOT / "rgpycrumbs").rglob("*.py")):
        text = path.read_text()
        pinned = re.search(r"^# chemparseplot = \{ git = ", text, re.M)
        for version in re.findall(
            r'^#.*"chemparseplot(?:\[[\w,]+\])?([^"@]*)"', text, re.M
        ):
            assert version == f">={floor},<2" or (not version and pinned), path
        assert 'branch = "main"' not in text, path
        floors = set(re.findall(r'"chemparseplot(?:\[[\w,]+\])?>=([\d.]+)', text))
        assert floors <= {floor}, path
    for path in sorted((PROJECT_ROOT / "docs" / "orgmode").rglob("*.org")):
        documented = re.findall(
            r"chemparseplot ?(?:≥|>=) ?(\d+(?:\.\d+)+)", path.read_text()
        )
        assert set(documented) <= {floor}, path


@pytest.mark.parametrize(
    ("purl", "pins"),
    [
        ("pkg:pypi/example%2dpackage@1.2.3", {"example-package": "1.2.3"}),
        ("pkg:pypi/torch@2.5.1%2Bcpu", {"torch": "2.5.1+cpu"}),
        ("pkg:pypi/typing%5fextensions@4.12.2", {"typing-extensions": "4.12.2"}),
    ],
)
def test_cyclonedx_purls_decode_every_percent_escape(purl: str, pins: dict) -> None:
    """Package URL escapes are case-insensitive and apply to versions too."""
    from rgpycrumbs.locks import pypi_pins_from_cyclonedx

    document = {"components": [{"purl": purl, "type": "library"}]}

    assert pypi_pins_from_cyclonedx(document) == pins


@pytest.mark.parametrize(
    ("name", "lines"),
    [
        ("values.txt", ["alpha", "beta", "gamma"]),
        ("values.txt.gz", ["alpha", "beta", "gamma"]),
        ("wide.txt.gz", [f"{'x' * 3000} {index}" for index in range(10)]),
    ],
)
def test_tail_returns_the_last_lines(tmp_path: Path, name: str, lines: list[str]) -> None:
    """Plain and gzip files give the last lines, however long the lines are."""
    from rgpycrumbs.search.helpers import tail

    path = tmp_path / name
    text = "".join(f"{line}\n" for line in lines)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="utf-8")

    assert tail(path, 4) == lines[-4:]


@pytest.mark.parametrize("content", ["", "one line\n"])
def test_head_search_returns_false_at_early_eof(tmp_path: Path, content: str) -> None:
    """A file shorter than the line budget is a plain miss."""
    from rgpycrumbs.search.helpers import head_search

    path = tmp_path / "short.txt"
    path.write_text(content, encoding="utf-8")

    assert head_search(path, "absent") is False


def test_head_search_reads_to_the_end_without_a_line_budget(tmp_path: Path) -> None:
    """Only an explicit ``n`` bounds the search."""
    from rgpycrumbs.search.helpers import head_search

    lines = [f"line {index}" for index in range(1000)] + ["needle"]
    path = tmp_path / "late.txt"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

    assert head_search(path, "needle") is True
    assert head_search(path, "needle", n=len(lines) - 1) is False


# The child sleeps for ten deadlines, so it is still running when the deadline
# passes even if this thread loses the CPU for most of that time.
LIVE_TIMEOUT_SECONDS = 0.5
SILENT_CHILD_SECONDS = 10 * LIVE_TIMEOUT_SECONDS


def test_live_command_timeout_stops_a_running_child() -> None:
    """The deadline stops a child that is still running, even a silent one."""
    from rgpycrumbs.run.jupyter import _run_command_live

    with pytest.raises(subprocess.TimeoutExpired):
        _run_command_live(
            [sys.executable, "-c", f"import time; time.sleep({SILENT_CHILD_SECONDS})"],
            timeout=LIVE_TIMEOUT_SECONDS,
        )


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX sessions")
@pytest.mark.parametrize(
    "script",
    ["{sleep} & {sleep}", "{sleep} &"],
    ids=["shell-still-running", "shell-exited"],
)
def test_live_command_timeout_kills_what_a_shell_command_starts(script: str) -> None:
    """A background child that keeps the output open ends at the deadline too."""
    from rgpycrumbs.run.jupyter import _run_command_live

    sleep = f"sleep {SILENT_CHILD_SECONDS:g}"
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_command_live(script.format(sleep=sleep), timeout=LIVE_TIMEOUT_SECONDS)
    assert time.monotonic() - start < SILENT_CHILD_SECONDS


def test_run_command_or_exit_sets_no_deadline_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a timeout argument the command runs as long as it needs."""
    from rgpycrumbs.run import jupyter

    seen = {}

    def run_live(cmd, *, check, capture, timeout):
        seen["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(jupyter, "_run_command_live", run_live)
    jupyter.run_command_or_exit(["true"])

    assert seen == {"timeout": None}


def test_cli_discovery_requires_pep_723_script_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only files with a PEP 723 script block become dispatcher commands."""
    from rgpycrumbs import cli

    group = tmp_path / "tools"
    group.mkdir()
    (group / "library.py").write_text('"""Run it with a # /// script header."""\n')
    (group / "command.py").write_text("# /// script\n# dependencies = []\n# ///\n")
    monkeypatch.setattr(cli, "PACKAGE_ROOT", tmp_path)

    assert cli._get_scripts_in_folder("tools") == ["command"]


def test_suite_pins_let_environment_pins_override_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment pins rank above config pins, as rgpycrumbs.config documents."""
    from rgpycrumbs.api import suite_pins

    (tmp_path / "rgpkgs.toml").write_text(
        '[pins.packages]\nnumpy = "1.26.4"\nscipy = "1.11.4"\n', encoding="utf-8"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in ("RGPKGS_CONFIG", "RGPYCRUMBS_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RGPKGS_LOCK_PINS", '{"numpy": "2.0.0"}')
    monkeypatch.chdir(tmp_path)

    assert suite_pins() == {"numpy": "2.0.0", "scipy": "1.11.4"}


def test_bond_order_fragments_report_a_missing_tblite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without tblite the bond-order method raises ImportError with install routes."""
    ase = pytest.importorskip("ase")
    from rgpycrumbs.geom import fragments

    monkeypatch.setattr(fragments, "tbliteinterface", None)
    hydrogen = ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    with pytest.raises(ImportError, match=r"needs tblite(?s:.)*pip install tblite"):
        fragments.find_fragments_bond_order(
            hydrogen, threshold=0.5, charge=0, multiplicity=1
        )


def test_one_day_tdelta_runs_without_the_python_311_utc_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper works with the datetime module of Python 3.10, which lacks UTC."""
    import datetime

    from rgpycrumbs.time import helpers

    python310_datetime = SimpleNamespace(
        datetime=datetime.datetime,
        timedelta=datetime.timedelta,
        timezone=datetime.timezone,
    )
    monkeypatch.setattr(helpers, "datetime", python310_datetime)

    assert helpers.one_day_tdelta("00:00:01", "23:59:59") == datetime.timedelta(seconds=2)


@pytest.mark.parametrize(
    ("x", "y", "options", "message"),
    [
        ([0.0, 2.0, 1.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0], {}, "strictly increasing"),
        ([0.0, 1.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0, 4.0], {}, "strictly increasing"),
        ([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0], {}, "differ in length"),
        ([[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]], {}, "one-dimensional"),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], {"knots": 3}, "needs more than 3 samples"),
        ([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0], {"num": 0}, "at least 1"),
    ],
    ids=["unordered", "repeated", "length", "two-dimensional", "too-few", "empty-grid"],
)
def test_spline_interp_rejects_inputs_it_cannot_fit(x, y, options, message) -> None:
    """Inputs that SciPy rejects or turns into NaN raise ValueError up front."""
    from rgpycrumbs.interpolation import spline_interp

    with pytest.raises(ValueError, match=message):
        spline_interp(np.asarray(x), np.asarray(y), **options)


def test_spline_interp_accepts_lists() -> None:
    """Lists give the same grid and values as arrays."""
    pytest.importorskip("scipy")
    from rgpycrumbs.interpolation import spline_interp

    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 1.0, 4.0, 9.0, 16.0]

    np.testing.assert_array_equal(
        spline_interp(x, y, num=9), spline_interp(np.asarray(x), np.asarray(y), num=9)
    )


# More samples than the 100 points the default grid used to have.
SPLINE_SAMPLES = 200


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_spline_interp_default_grid_passes_through_every_sample(degree: int) -> None:
    """Without num, the grid of an evenly spaced input contains every sample."""
    pytest.importorskip("scipy")
    from rgpycrumbs.interpolation import spline_interp

    x = np.linspace(1.0, 2.0, SPLINE_SAMPLES)
    y = x**2

    x_fine, y_fine = spline_interp(x, y, knots=degree)

    np.testing.assert_allclose(x_fine[::degree], x)
    np.testing.assert_allclose(y_fine[::degree], y)


def test_muller_brown_parser_skips_iterations_with_nan() -> None:
    """Iteration lines with nan coordinates are dropped and the rest stacked."""
    pytest.importorskip("cmcrameri")
    from rgpycrumbs.xts.saddle.mb import parse_iteration_paths

    lines = [
        "log header",
        "Iteration 0: [0.5, 0.1]",
        "Iteration 1: [nan, nan]",
        "Iteration 2: [0.4, 0.2]",
    ]

    np.testing.assert_array_equal(parse_iteration_paths(lines), [[0.5, 0.1], [0.4, 0.2]])


def test_muller_brown_script_reports_a_log_without_iterations(tmp_path: Path) -> None:
    """The plot script names the missing Iteration lines instead of failing in NumPy."""
    pytest.importorskip("cmcrameri")
    log = tmp_path / "output.txt"
    log.write_text("log header\nIteration 1: [nan, nan]\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rgpycrumbs.xts.saddle.mb",
            "--nimgs",
            "1",
            "--ifile",
            str(log),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "no Iteration line without nan coordinates" in result.stderr
