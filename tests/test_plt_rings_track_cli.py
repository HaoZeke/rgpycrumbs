# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Dispatcher route for the primitive-ring trajectory."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner


def test_plt_rings_track_declares_the_wheel():
    root = Path(__file__).resolve().parent.parent
    script = root / "rgpycrumbs" / "geom" / "plt_rings_track.py"
    text = script.read_text()
    assert '# requires-python = ">=3.12"' in text
    assert "pydseamslib>=2.9.1" in text
    assert "chemparseplot>=1.11" in text


@patch("rgpycrumbs.cli.subprocess.run")
def test_plt_rings_track_help_routes_through_dispatcher(mock_run, monkeypatch):
    from rgpycrumbs.cli import main

    monkeypatch.setattr("rgpycrumbs.cli.Path.is_file", lambda self: True)
    monkeypatch.setattr("rgpycrumbs.cli._uv_editable_sources", lambda: [])

    result = CliRunner().invoke(main, ["geom", "plt-rings-track", "--help"])
    assert result.exit_code == 0, result.exception or result.output

    command = mock_run.call_args.args[0]
    assert any(str(part).endswith("geom/plt_rings_track.py") for part in command)
    assert command[-1] == "--help"
