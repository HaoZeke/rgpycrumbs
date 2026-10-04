#!/usr/bin/env python3
"""Draw the band of a surrogate-assisted NEB: profile, evolution, landscape.

.. versionadded:: 1.12.0

Reads one search directory with a chemparseplot parser and draws

- ``profile``: surrogate mean along the path, uncertainty ribbon when the
  producer recorded one, the true energies at the images and the oracle
  evaluations projected onto the path (``--observation-mode`` says how their
  distance from the path is shown);
- ``evolution``: the band over outer iterations (snapshots), or the record of
  which image the oracle was called on when only the final band exists;
- ``landscape``: the reaction-valley landscape of ``rgpycrumbs eon plt-neb``
  (progress ``s`` against orthogonal deviation ``d``, drawn by the same
  chemparseplot functions) with the GP energy surface fitted to the oracle
  evaluations, the evaluations as dots, the final path, the climbing image and
  the reported saddle, each in the legend. Needs ``jax`` (in the script
  metadata).

Prefer the dispatcher::

    rgpycrumbs surrogate plt-band results/rec/baker/25_hcnh2 -o figs/hcnh2
    rgpycrumbs surrogate plt-band run/ --producer ml-neb --panel profile

Files are ``<prefix>-profile``, ``-evolution`` and ``-landscape`` (the last
only when the producer kept the geometries of its evaluations). Output is
byte-reproducible; each file embeds the SHA-256 of the inputs and the tool
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
#   "scipy>=1.11",
#   "jax>=0.4",
#   "polars>=0.20",
#   "rgpycrumbs>=1.10",
#   "xyzrender>=0.3.8",
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@f57f5e9a5e35c38e227cebfbc900444e29389df2",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate import read_search
from chemparseplot.parse.surrogate.gpr_optim import read_atom_types
from chemparseplot.plot.provenance import file_sha256, save_with_provenance
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
    "--landscape-surface/--no-landscape-surface",
    "landscape_surface",
    default=True,
    show_default=True,
    help="Draw the GP energy surface fitted to the oracle evaluations.",
)
@click.option(
    "--landscape-color",
    type=click.Choice(["energy", "iteration"]),
    default="energy",
    show_default=True,
    help="Colour of the oracle evaluations: true energy or order of evaluation.",
)
@click.option(
    "--landscape-fade-variance",
    type=float,
    default=0.95,
    show_default=True,
    help="Fade the GP surface where its relative variance exceeds this level (0-1, the outermost labelled contour); 1 or more keeps all of it.",
)
@click.option(
    "--landscape-label-every",
    type=int,
    default=None,
    help="Number every K-th oracle evaluation (order of evaluation); default none.",
)
@click.option(
    "--plot-structures",
    type=click.Choice(["crit_points", "all", "none"]),
    default="crit_points",
    show_default=True,
    help="Strip of structures under the profile and the landscape, as rgpycrumbs eon plt-neb draws: reactant, saddle (or climbing image) and product, every band image, or none.",
)
@click.option(
    "--n-structures",
    type=int,
    default=None,
    help="With crit_points: draw this many images, evenly spaced and always including reactant, saddle and product.",
)
@click.option(
    "--types-from",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Geometry file (any ASE-readable file, or a .con) with the atoms in the band's order; supplies the elements for the strip when the cell holds none.",
)
@click.option(
    "--strip-renderer",
    type=click.Choice(["xyzrender", "ase", "solvis", "ovito"]),
    default="xyzrender",
    show_default=True,
    help="Structure renderer of the strip.",
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
    plot_structures,
    n_structures,
    types_from,
    strip_renderer,
    landscape_surface,
    landscape_color,
    landscape_fade_variance,
    landscape_label_every,
    observation_mode,
    observation_distance,
):
    """Draw the band profile, its evolution and the reaction-valley landscape."""
    set_font(font, font_dirs)
    s = read_search(search, producer)
    ttl = s.label if title is None else title
    if s.band is None:
        raise click.ClickException(f"{search} holds no band (no band.h5 or trajectories)")
    hashes = dict(s.provenance)
    if types_from is not None:
        try:
            s.band.numbers = read_atom_types(
                types_from, s.band.final.positions.shape[1] // 3
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        s.band.numbers_source = str(types_from)
        hashes[f"types:{types_from.name}"] = file_sha256(types_from)
    strip = {
        "structures": None if plot_structures == "none" else plot_structures,
        "n_structures": n_structures,
        "strip_renderer": strip_renderer,
    }
    command = " ".join(["rgpycrumbs", "surrogate", "plt-band", *sys.argv[1:]])
    has_points = s.band.points is not None and s.band.points.positions is not None
    if panels and "landscape" in panels and not has_points:
        raise click.ClickException(f"{search} kept no geometries of its observations")
    wanted = panels or ("profile", "evolution", *(("landscape",) if has_points else ()))
    for panel in wanted:
        try:
            if panel == "profile":
                fig = plot_band_profile(
                    s.band,
                    energy_unit=energy_unit,
                    title=ttl,
                    observations=observation_mode if profile_observations else "none",
                    observation_distance=observation_distance,
                    **strip,
                )
            elif panel == "evolution":
                fig = plot_band_evolution(s.band, energy_unit=energy_unit)
            else:
                fig = plot_reduced_landscape(
                    s.band,
                    energy_unit=energy_unit,
                    title=ttl,
                    surface="grad_matern" if landscape_surface else None,
                    color_by=landscape_color,
                    fade_variance=landscape_fade_variance
                    if landscape_fade_variance < 1
                    else None,
                    label_every=landscape_label_every,
                    **strip,
                )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        out = prefix.with_name(f"{prefix.name}-{panel}.{fmt}")
        save_with_provenance(fig, out, {}, command=command, hashes=hashes, dpi=dpi)
        click.echo(out)


if __name__ == "__main__":
    main()
