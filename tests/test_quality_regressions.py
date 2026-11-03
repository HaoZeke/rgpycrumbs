"""Regression tests, one or more per fixed defect."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
from pathlib import Path
from types import SimpleNamespace

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
