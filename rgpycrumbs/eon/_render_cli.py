"""Shared Click option bundles for eOn structure-rendering CLIs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

F = TypeVar("F", bound=Callable[..., object])

RENDERER_CHOICES = ("ase", "xyzrender", "solvis", "ovito")

CONFIG_HELP = (
    "TOML plot config. Shared style/render settings live under [shared]; "
    "command inputs under [neb], [min], or [saddle]. Explicit CLI flags override "
    "the file. Example:\n\n"
    "  [shared]\n"
    "  energy_unit = \"eV\"\n"
    "  dpi = 200\n"
    "  strip_renderer = \"xyzrender\"\n\n"
    "  [neb]\n"
    "  con_file = \"neb.con\"\n"
    "  plot_type = \"landscape\"\n"
    "  output_file = \"neb.pdf\"\n"
)


def _apply(options: list[Callable[[F], F]], func: F) -> F:
    """Apply Click decorators in source order."""

    for option in reversed(options):
        func = option(func)
    return func


def add_config_option(func: F) -> F:
    """Attach ``--config`` for declarative TOML plot settings."""

    return click.option(
        "--config",
        "config",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help=CONFIG_HELP,
    )(func)


def add_render_options(func: F) -> F:
    """Attach the shared renderer-related CLI options to a command."""

    options = [
        click.option(
            "--strip-renderer",
            type=click.Choice(list(RENDERER_CHOICES)),
            default="xyzrender",
            show_default=True,
            help=(
                "Rendering backend for structure images. "
                "xyzrender/ase work with the default dispatcher setup; "
                "solvis and ovito require separate heavy installs. "
                "Prefer [shared] strip_renderer in --config for suites."
            ),
        ),
        click.option(
            "--xyzrender-config",
            type=str,
            default="paton",
            show_default=True,
            help="xyzrender preset (paton, bubble, flat, tube, wire, skeletal).",
        ),
        click.option(
            "--strip-spacing",
            type=float,
            default=1.5,
            show_default=True,
            help="Horizontal spacing between structure images.",
        ),
        click.option(
            "--strip-dividers/--no-strip-dividers",
            is_flag=True,
            default=False,
            help="Draw vertical divider lines between structure images.",
        ),
        click.option(
            "--rotation",
            type=str,
            default="auto",
            help=(
                "Viewing rotation. 'auto' lets xyzrender auto-orient (default). "
                "ASE-style string e.g. '0x,90y,0z' for manual control."
            ),
        ),
        click.option(
            "--perspective-tilt",
            type=float,
            default=0.0,
            show_default=True,
            help="Small off-axis tilt (degrees) to reveal occluded atoms. 5-10 is typical.",
        ),
    ]
    return _apply(options, func)
