#!/usr/bin/env python3
"""Draw potential energy on a metric 2D plane (MethodsX).

Reads ``# x y energy [f_para] [step]`` (``chemparseplot.energy.v1``)
and calls ``chemparseplot.plot.representation.plot_energy``. Occupancy
FES CSVs are not this field.

.. versionadded:: 1.10.11
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "jax",
#   "chemparseplot[neb,plot]>=1.9.17,<2",
#   "rgpycrumbs>=1.10.4",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import click

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs landfold plot-energy")


@click.command()
@click.option(
    "--input",
    "inp",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Energy table: x y energy [f_para] [step].",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output figure path (.png or .svg).",
)
@click.option(
    "--frame",
    type=click.Choice(["plane", "rmsd", "progress", "landfold"]),
    default="rmsd",
    show_default=True,
    help="rmsd rotates (r,p) to (s,d). landfold keeps (s1,s2).",
)
@click.option(
    "--project-path/--no-project-path",
    default=None,
    help="Override frame default for the (s,d) rotation.",
)
@click.option("--clabel", default=r"$E$", show_default=True)
@click.option("--xlabel", default=None)
@click.option("--ylabel", default=None)
@click.option(
    "--method",
    default="grad_imq",
    show_default=True,
    help="rgpycrumbs.surfaces model (grad_imq matches eOn NEB TOML).",
)
@click.option("--auto-thin/--no-auto-thin", default=True, show_default=True)
@click.option("--max-surface-points", default=300, show_default=True, type=int)
@click.option("--n-inducing", default=None, type=int)
@click.option(
    "--rbf-smooth",
    default=None,
    type=float,
    help="IMQ/Matern length hint. Default is 0.1 of the plane span.",
)
@click.option("--show-pts/--hide-pts", default=True, show_default=True)
def main(
    inp,
    output,
    frame,
    project_path,
    clabel,
    xlabel,
    ylabel,
    method,
    auto_thin,
    max_surface_points,
    n_inducing,
    rbf_smooth,
    show_pts,
) -> None:
    from chemparseplot.parse.representation import from_path_forces, load_energy_table
    from chemparseplot.plot.neb import SurfaceFitConfig
    from chemparseplot.plot.representation import plot_energy

    rep = load_energy_table(inp, frame=frame)
    if rep.f_para is not None and (rep.grad_x is None or rep.grad_y is None):
        rep = from_path_forces(
            rep.x,
            rep.y,
            rep.energy,
            rep.f_para,
            frame=frame,
            step=rep.step,
        )
    fig = plot_energy(
        rep,
        project_path=project_path,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=clabel,
        method=method,
        surface_fit=SurfaceFitConfig(
            auto_thin=auto_thin, max_surface_points=max_surface_points
        ),
        n_inducing=n_inducing,
        rbf_smooth=rbf_smooth,
        show_pts=show_pts,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    click.echo(str(output))


if __name__ == "__main__":
    main()
