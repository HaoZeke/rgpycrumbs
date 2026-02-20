import sys
import pytest
from unittest.mock import patch
from click.testing import CliRunner
from rgpycrumbs.cli import main, _make_script_command

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

    # We also need to mock the path resolution inside _dispatch so it doesn't sys.exit(1)
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
    assert "dummy_script.py" in str(executed_command[2])
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
