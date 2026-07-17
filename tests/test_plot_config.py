# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for declarative eOn plot TOML config merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from rgpycrumbs.eon.plot_config import (
    MINIMAL_CONFIG_EXAMPLE,
    SHARED_DEFAULTS,
    extract_config_layers,
    load_plot_config,
    merge_plot_settings,
    resolve_from_click,
)


def test_minimal_example_parses_and_has_shared_keys(tmp_path: Path):
    path = tmp_path / "plot.toml"
    path.write_text(MINIMAL_CONFIG_EXAMPLE)
    data = load_plot_config(path)
    shared, neb = extract_config_layers(data, "neb")
    assert shared["energy_unit"] == "eV"
    assert shared["dpi"] == 200
    assert shared["strip_renderer"] == "xyzrender"
    assert neb["con_file"] == Path("neb.con")
    assert neb["plot_type"] == "landscape"
    assert neb["output_file"] == Path("neb_landscape.pdf")


def test_merge_shared_and_command_layers(tmp_path: Path):
    path = tmp_path / "plot.toml"
    path.write_text(
        """
[shared]
energy_unit = "kcal/mol"
dpi = 150
figsize = [6.0, 4.0]
strip_renderer = "ase"

[neb]
con_file = "band.con"
plot_type = "profile"
output_file = "out.pdf"
title = "From config"

[min]
job_dir = ["run_a", "run_b"]
label = ["A", "B"]
plot_type = "landscape"
output = "min.pdf"
"""
    )
    neb = merge_plot_settings("neb", config_path=path)
    assert neb["energy_unit"] == "kcal/mol"
    assert neb["dpi"] == 150
    assert neb["figsize"] == (6.0, 4.0)
    assert neb["strip_renderer"] == "ase"
    assert neb["con_file"] == Path("band.con")
    assert neb["title"] == "From config"
    # command defaults still present
    assert neb["source"] == "eon"
    assert neb["theme"] == SHARED_DEFAULTS["theme"]

    min_settings = merge_plot_settings("min", config_path=path)
    assert min_settings["job_dir"] == (Path("run_a"), Path("run_b"))
    assert min_settings["label"] == ("A", "B")
    assert min_settings["energy_unit"] == "kcal/mol"
    assert min_settings["output"] == Path("min.pdf")
    assert min_settings["surface_type"] == "grad_matern"


def test_cli_overrides_beat_config(tmp_path: Path):
    path = tmp_path / "plot.toml"
    path.write_text(
        """
[shared]
energy_unit = "eV"
dpi = 100

[neb]
output_file = "from_config.pdf"
"""
    )
    settings = merge_plot_settings(
        "neb",
        config_path=path,
        cli_overrides={"energy_unit": "kJ/mol", "dpi": 300, "output_file": "cli.pdf"},
    )
    assert settings["energy_unit"] == "kJ/mol"
    assert settings["dpi"] == 300
    assert settings["output_file"] == Path("cli.pdf")


def test_flat_top_level_shared_keys(tmp_path: Path):
    path = tmp_path / "flat.toml"
    path.write_text(
        """
energy_unit = "kcal/mol"
dpi = 120
con_file = "x.con"
plot_type = "landscape"
"""
    )
    settings = merge_plot_settings("neb", config_path=path)
    assert settings["energy_unit"] == "kcal/mol"
    assert settings["dpi"] == 120
    assert settings["con_file"] == Path("x.con")
    assert settings["plot_type"] == "landscape"


def test_unknown_command_raises():
    with pytest.raises(ValueError, match="Unknown plot command"):
        merge_plot_settings("kmc")


def test_example_file_in_package():
    example = (
        Path(__file__).resolve().parents[1]
        / "rgpycrumbs"
        / "eon"
        / "examples"
        / "plot_config.example.toml"
    )
    assert example.is_file()
    data = load_plot_config(example)
    assert "shared" in data and "neb" in data and "min" in data


def test_resolve_from_click_uses_parameter_source(tmp_path: Path):
    import click
    from click.testing import CliRunner

    config = tmp_path / "c.toml"
    config.write_text(
        """
[shared]
energy_unit = "kcal/mol"
dpi = 111

[min]
job_dir = ["from_config"]
plot_type = "convergence"
"""
    )

    @click.command()
    @click.pass_context
    @click.option("--config", type=click.Path(path_type=Path), default=None)
    @click.option("--energy-unit", default="eV")
    @click.option("--dpi", type=int, default=200)
    @click.option("--job-dir", multiple=True, default=None)
    @click.option("--plot-type", default="profile")
    def fake(ctx, config, energy_unit, dpi, job_dir, plot_type):
        settings = resolve_from_click(
            "min",
            ctx,
            config=config,
            energy_unit=energy_unit,
            dpi=dpi,
            job_dir=job_dir,
            plot_type=plot_type,
        )
        click.echo(
            f"{settings['energy_unit']}|{settings['dpi']}|{settings['plot_type']}|"
            f"{list(settings['job_dir'])}"
        )

    runner = CliRunner()
    # config only
    result = runner.invoke(fake, ["--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "kcal/mol|111|convergence|" in result.output
    assert "from_config" in result.output

    # CLI override energy + dpi
    result2 = runner.invoke(
        fake, ["--config", str(config), "--energy-unit", "kJ/mol", "--dpi", "9"]
    )
    assert result2.exit_code == 0, result2.output
    assert result2.output.startswith("kJ/mol|9|convergence|")


def test_cli_help_mentions_config():
    import ast
    from pathlib import Path

    import click
    from click.testing import CliRunner

    from rgpycrumbs.eon._render_cli import CONFIG_HELP, add_config_option

    @click.command()
    @add_config_option
    def dummy():
        """dummy"""

    runner = CliRunner()
    result = runner.invoke(dummy, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--config" in result.output
    assert "TOML" in result.output
    assert "[shared]" in CONFIG_HELP

    eon = Path(__file__).resolve().parents[1] / "rgpycrumbs" / "eon"
    for name in ("plt_neb.py", "plt_min.py", "plt_saddle.py"):
        tree = ast.parse((eon / name).read_text(), filename=name)
        src = (eon / name).read_text()
        assert "add_config_option" in src
        assert "run_from_click" in src
        assert "library_plot" in src
        assert "@click.pass_context" in src or "pass_context" in src


def test_surface_fit_keys_from_toml_default_off(tmp_path: Path):
    path = tmp_path / "plot.toml"
    path.write_text(
        """
[shared]
auto_thin = false
max_surface_points = 64

[min]
job_dir = ["run"]
plot_type = "landscape"
surface_type = "grad_imq"
"""
    )
    settings = merge_plot_settings("min", config_path=path)
    assert settings["auto_thin"] is False
    assert settings["max_surface_points"] == 64
    assert settings["surface_type"] == "grad_imq"


def test_surface_fit_keys_opt_in_via_min_section(tmp_path: Path):
    path = tmp_path / "plot.toml"
    path.write_text(
        """
[min]
job_dir = ["run"]
plot_type = "landscape"
auto_thin = true
max_surface_points = 48
"""
    )
    settings = merge_plot_settings("min", config_path=path)
    assert settings["auto_thin"] is True
    assert settings["max_surface_points"] == 48
