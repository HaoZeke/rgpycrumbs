# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Dispatcher route for the primitive-ring figure."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner


def test_plt_rings_targets_the_published_wheel():
    root = Path(__file__).resolve().parent.parent
    script = root / "rgpycrumbs" / "geom" / "plt_rings.py"
    text = script.read_text()
    assert '# requires-python = ">=3.12"' in text
    assert "pydseamslib>=2.9.1" in text
    assert "xyzrender>=0.3.8" in text


@patch("rgpycrumbs.cli.subprocess.run")
def test_plt_rings_help_routes_through_dispatcher(mock_run, monkeypatch):
    from rgpycrumbs.cli import main

    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setattr("rgpycrumbs.cli._uv_editable_sources", lambda: [])

    result = CliRunner().invoke(main, ["geom", "plt-rings", "--help"])
    assert result.exit_code == 0, result.exception or result.output

    command = mock_run.call_args.args[0]
    assert any(str(part).endswith("geom/plt_rings.py") for part in command)
    assert command[-1] == "--help"
