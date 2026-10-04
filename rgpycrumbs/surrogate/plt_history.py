#!/usr/bin/env python3
"""Draw convergence and model diagnostics of a surrogate-assisted search.

.. versionadded:: 1.12.0

For one search directory:

- ``convergence``: max atomic force against oracle calls (true force over the
  evaluated images, the surrogate's prediction, the climbing image, the
  tolerance), acquisition decisions below the axis, training-set size below;
- ``diagnostics``: kernel amplitude and noise, length scales, training rows
  and seconds per outer iteration.

With ``--against OTHER`` the two searches are compared on one set of axes
with a stacked ledger of oracle calls by phase, which is the view that shows
where one run spent calls the other did not::

    rgpycrumbs surrogate plt-history rec_new/baker/25_hcnh2 -o figs/h
    rgpycrumbs surrogate plt-history rec_old/baker/25_hcnh2 --against rec_new/baker/25_hcnh2 -o figs/hcnh2-regression
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@34ec7c94b2ae5fd3a54baaa31f4cffb15a8ae72e",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate import read_search
from chemparseplot.plot.provenance import save_with_provenance
from chemparseplot.plot.surrogate import (
    plot_model_diagnostics,
    plot_search_comparison,
    plot_search_convergence,
    set_font,
)

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-history")


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
)
@click.option(
    "--against",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="A second search to compare with (same producer).",
)
@click.option("--label", default=None, help="Legend label for SEARCH.")
@click.option("--against-label", default=None, help="Legend label for --against.")
@click.option(
    "--x",
    "xaxis",
    type=click.Choice(["oracle_calls", "outer"]),
    default="oracle_calls",
    show_default=True,
    help="Horizontal axis of the convergence panel.",
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
    against,
    label,
    against_label,
    xaxis,
    fmt,
    dpi,
    font,
    font_dirs,
):
    """Draw convergence and diagnostics for SEARCH."""
    set_font(font, font_dirs)
    s = read_search(search, producer)
    if s.search is None:
        raise click.ClickException(f"{search} holds no per-iteration history")
    command = " ".join(["rgpycrumbs", "surrogate", "plt-history", *sys.argv[1:]])
    figs = {
        "convergence": plot_search_convergence(s.search, x=xaxis, title=label or s.label),
        "diagnostics": plot_model_diagnostics(s.search, x="outer"),
    }
    hashes = dict(s.provenance)
    if against is not None:
        other = read_search(against, producer)
        figs["comparison"] = plot_search_comparison(
            [other, s], [against_label or other.label, label or s.label]
        )
        hashes.update({f"against:{k}": v for k, v in other.provenance.items()})
    for name, fig in figs.items():
        out = prefix.with_name(f"{prefix.name}-{name}.{fmt}")
        save_with_provenance(fig, out, {}, command=command, hashes=hashes, dpi=dpi)
        click.echo(out)


if __name__ == "__main__":
    main()
