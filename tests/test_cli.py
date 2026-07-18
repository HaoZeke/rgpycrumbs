import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from rgpycrumbs.cli import _dispatch, _make_script_command, main

pytestmark = pytest.mark.pure


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_script_group(monkeypatch):
    """Mock the script discovery so the CLI has a command to run."""
    # Temporarily add a dummy command to the main group for testing
    dummy_cmd = _make_script_command("dummy_group", "dummy_script.py")
    main.add_command(dummy_cmd, name="dummy_script")

    # We also need to mock the path resolution inside _dispatch so it doesn't
    # sys.exit(1)
    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setattr("rgpycrumbs.cli.Path.resolve", lambda self: self)


@patch("rgpycrumbs.cli.subprocess.run")
def test_cli_standard_execution(mock_run, runner, mock_script_group):
    """Test that default execution uses 'uv run'."""
    result = runner.invoke(main, ["dummy_script", "arg1"])

    assert result.exit_code == 0

    # Extract the command list passed to the mocked execution function
    executed_command = mock_run.call_args[0][0]

    assert executed_command[0] == "uv"
    assert executed_command[1] == "run"
    assert any("dummy_script.py" in str(part) for part in executed_command)
    assert "arg1" in executed_command


@patch("rgpycrumbs.cli.subprocess.run")
def test_cli_dev_execution(mock_run, runner, mock_script_group):
    """Test that the --dev flag switches execution to sys.executable."""
    result = runner.invoke(main, ["--dev", "dummy_script", "arg1"])

    assert result.exit_code == 0

    executed_command = mock_run.call_args[0][0]

    # Verify 'uv run' was bypassed in favor of the active Python interpreter
    assert executed_command[0] == sys.executable
    assert "dummy_script.py" in str(executed_command[1])


@patch("rgpycrumbs.cli.subprocess.run")
def test_cli_verbose_output(mock_run, runner, mock_script_group):
    """Test that the --verbose flag prints the paths before execution."""
    result = runner.invoke(main, ["--verbose", "dummy_script"])

    assert result.exit_code == 0
    # Check that our verbose click.echo statements fired
    assert "VERBOSE: Resolved script path ->" in result.output
    assert "VERBOSE: Constructed command -> uv run" in result.output


@patch("rgpycrumbs.cli.subprocess.run")
def test_dispatch_preserves_user_site_package_path(mock_run, monkeypatch):
    """_dispatch should join parent site-packages paths, not split strings."""
    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setattr(
        "rgpycrumbs.cli.site.getsitepackages",
        lambda: ["/global/site-packages"],
    )
    monkeypatch.setattr(
        "rgpycrumbs.cli.site.getusersitepackages",
        lambda: "/user/site-packages",
    )

    _dispatch("group", "script", ())

    env = mock_run.call_args.kwargs["env"]
    assert env["RGPYCRUMBS_PARENT_SITE_PACKAGES"] == (
        "/global/site-packages:/user/site-packages"
    )


@patch("rgpycrumbs.cli.subprocess.run")
def test_dispatch_adds_editable_sources_for_linked_packages(
    mock_run, monkeypatch, tmp_path
):
    """_dispatch should satisfy local linked deps via --with-editable."""
    # True editable: checkout outside site-packages with matching project name.
    root = tmp_path / "chemparseplot"
    (root / "chemparseplot").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "chemparseplot"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    init = root / "chemparseplot" / "__init__.py"
    init.write_text("# editable\n", encoding="utf-8")

    # Only pretend the *script* path exists; keep real is_file for pyproject walk.
    real_is_file = Path.is_file

    def _is_file(self: Path) -> bool:
        s = str(self)
        if s.endswith("script.py") or s.endswith("group/script.py"):
            return True
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", _is_file)
    monkeypatch.setattr(
        "rgpycrumbs.cli.importlib.util.find_spec",
        lambda name: (
            SimpleNamespace(
                origin=str(init),
                submodule_search_locations=[str(init.parent)],
            )
            if name == "chemparseplot"
            else None
        ),
    )

    _dispatch("group", "script", ("--flag",))

    executed_command = mock_run.call_args[0][0]
    assert executed_command[:2] == ["uv", "run"]
    assert "--with-editable" in executed_command
    editable_idx = executed_command.index("--with-editable")
    assert Path(executed_command[editable_idx + 1]) == root
    assert str(executed_command[-2]).endswith("script.py")
    assert executed_command[-1] == "--flag"


@patch("rgpycrumbs.cli.subprocess.run")
def test_dispatch_skips_editable_sources_when_absent(mock_run, monkeypatch):
    """_dispatch should not add editable flags without a linked checkout."""
    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setattr("rgpycrumbs.cli.importlib.util.find_spec", lambda name: None)

    _dispatch("group", "script", ())

    executed_command = mock_run.call_args[0][0]
    assert executed_command[:2] == ["uv", "run"]
    assert "--with-editable" not in executed_command


@patch("rgpycrumbs.cli.subprocess.run")
def test_dispatch_defaults_auto_deps(mock_run, monkeypatch):
    """CLI dispatch enables ensure_import auto-install unless user set it."""
    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.delenv("RGPYCRUMBS_AUTO_DEPS", raising=False)

    _dispatch("group", "script", ())

    env = mock_run.call_args.kwargs["env"]
    assert env["RGPYCRUMBS_AUTO_DEPS"] == "1"


@patch("rgpycrumbs.cli.subprocess.run")
def test_dispatch_respects_explicit_auto_deps_off(mock_run, monkeypatch):
    """Explicit RGPYCRUMBS_AUTO_DEPS=0 must not be overridden by dispatch."""
    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setenv("RGPYCRUMBS_AUTO_DEPS", "0")

    _dispatch("group", "script", ())

    env = mock_run.call_args.kwargs["env"]
    assert env["RGPYCRUMBS_AUTO_DEPS"] == "0"


@patch("rgpycrumbs.cli.subprocess.run")
def test_dispatch_force_uv_uses_uv_run(mock_run, monkeypatch):
    """RGPYCRUMBS_FORCE_UV=1 must prefer uv run even if stack could be in-env."""
    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setenv("RGPYCRUMBS_FORCE_UV", "1")
    monkeypatch.setattr("rgpycrumbs.cli._in_env_stack_ready", lambda: True)
    monkeypatch.setattr("rgpycrumbs.cli.shutil.which", lambda name: "/usr/bin/uv")

    _dispatch("group", "script", ())

    executed = mock_run.call_args[0][0]
    assert executed[:2] == ["uv", "run"]


@patch("rgpycrumbs.cli.subprocess.run")
def test_dispatch_dev_uses_active_interpreter(mock_run, monkeypatch):
    """--dev / is_dev uses sys.executable (in-env), not uv."""
    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setenv("RGPYCRUMBS_FORCE_UV", "0")

    _dispatch("group", "script", (), is_dev=True)

    executed = mock_run.call_args[0][0]
    assert executed[0] == sys.executable


def test_config_show_command():
    """rgpycrumbs config show runs without error and prints lock_path key."""
    from click.testing import CliRunner

    from rgpycrumbs.cli import main

    result = CliRunner().invoke(main, ["config", "show"])
    assert result.exit_code == 0
    assert "lock_path:" in result.output
    assert "package_pins:" in result.output
