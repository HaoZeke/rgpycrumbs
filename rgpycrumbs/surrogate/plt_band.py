#!/usr/bin/env python3
"""Draw the band of a surrogate-assisted NEB: profile, evolution, landscape.

.. versionadded:: 1.12.0

Reads one search directory with a chemparseplot parser and draws

- ``profile``: surrogate mean along the path, uncertainty ribbon when the
  producer recorded one, true evaluations, retained observations;
- ``evolution``: the band over outer iterations (snapshots), or the record of
  which image the oracle was called on when only the final band exists.

Prefer the dispatcher::

    rgpycrumbs surrogate plt-band results/rec/baker/25_hcnh2 -o figs/hcnh2
    rgpycrumbs surrogate plt-band run/ --producer ml-neb --panel profile

- ``landscape``: the retained observations in the (s, d) reaction-valley
  plane, coloured by energy and numbered in acquisition order, with the band.

Files are ``<prefix>-profile``, ``-evolution`` and ``-landscape`` (the last
only when the producer kept geometries of the observations). Output
is byte-reproducible; each file embeds the SHA-256 of the inputs and the tool
versions, and a ``.provenance.json`` sits beside it.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click>=8.1",
#   "numpy>=1.26",
#   "matplotlib>=3.8.2",
#   "pint>=0.22",
#   "h5py>=3.0",
#   "ase>=3.22",
#   "pandas>=2.0",
#   "cmcrameri>=1.7",
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@9b04270ae772c342714ada96ad36ea5ad11b2576",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate import read_search
from chemparseplot.plot.provenance import save_with_provenance
from chemparseplot.plot.surrogate import (
    plot_band_evolution,
    plot_band_profile,
    plot_reduced_landscape,
    set_font,
)

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-band")


@click.command()
@click.argument("search", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; the panel name and format are appended.",
)
@click.option(
    "--producer",
    type=click.Choice(["gpr_optim", "ml-neb"]),
    default="gpr_optim",
    show_default=True,
    help="Which code wrote SEARCH.",
)
@click.option(
    "--panel",
    "panels",
    multiple=True,
    type=click.Choice(["profile", "evolution", "landscape"]),
    help="Panel to draw (repeatable). Default: all the record supports.",
)
@click.option(
    "--energy-unit",
    default="eV",
    show_default=True,
    help="eV, kcal/mol, kJ/mol or hartree.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    show_default=True,
)
@click.option("--dpi", type=int, default=200, show_default=True)
@click.option(
    "--profile-observations/--no-profile-observations",
    "profile_observations",
    default=True,
    show_default=True,
    help="Draw the oracle evaluations on the profile (they stay in the landscape).",
)
@click.option(
    "--observation-mode",
    type=click.Choice(["fade", "near"]),
    default="fade",
    show_default=True,
    help="fade: colour each evaluation by its distance from the final path; near: draw only those within --observation-distance and count the rest in the legend.",
)
@click.option(
    "--observation-distance",
    type=float,
    default=0.1,
    show_default=True,
    help="Cartesian distance (angstrom) for --observation-mode near.",
)
@click.option(
    "--landscape-labels/--no-landscape-labels",
    "landscape_labels",
    default=True,
    show_default=True,
    help="Number the evaluations on the landscape in order of evaluation.",
)
@click.option(
    "--title",
    default=None,
    help="Panel title; default the search label, '' for none.",
)
@click.option(
    "--font-dir",
    "font_dirs",
    multiple=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Extra directory searched for the --font faces (repeatable).",
)
@click.option(
    "--font",
    default=None,
    help="Font family for text and math text (e.g. Jost); an unresolvable family is an error. Default: the theme font.",
)
def main(
    *,
    search,
    prefix,
    producer,
    panels,
    energy_unit,
    fmt,
    dpi,
    font,
    font_dirs,
    title,
    profile_observations,
    landscape_labels,
    observation_mode,
    observation_distance,
):
    """Draw the band profile, its evolution and the observation landscape."""
    set_font(font, font_dirs)
    s = read_search(search, producer)
    ttl = s.label if title is None else title
    if s.band is None:
        raise click.ClickException(f"{search} holds no band (no band.h5 or trajectories)")
    command = " ".join(["rgpycrumbs", "surrogate", "plt-band", *sys.argv[1:]])
    has_points = s.band.points is not None and s.band.points.positions is not None
    if panels and "landscape" in panels and not has_points:
        raise click.ClickException(f"{search} kept no geometries of its observations")
    wanted = panels or ("profile", "evolution", *(("landscape",) if has_points else ()))
    for panel in wanted:
        if panel == "profile":
            fig = plot_band_profile(
                s.band,
                energy_unit=energy_unit,
                title=ttl,
                observations=observation_mode if profile_observations else "none",
                observation_distance=observation_distance,
            )
        elif panel == "evolution":
            fig = plot_band_evolution(s.band, energy_unit=energy_unit)
        else:
            fig = plot_reduced_landscape(
                s.band,
                energy_unit=energy_unit,
                title=ttl,
                label_numbers=landscape_labels,
            )
        out = prefix.with_name(f"{prefix.name}-{panel}.{fmt}")
        save_with_provenance(fig, out, {}, command=command, hashes=s.provenance, dpi=dpi)
        click.echo(out)


if __name__ == "__main__":
    main()
