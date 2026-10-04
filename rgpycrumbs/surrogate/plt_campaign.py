#!/usr/bin/env python3
"""Draw a campaign: calls per reaction, pass matrix, wall time.

.. versionadded:: 1.12.0

RECORD holds ``<set>/<cell>/result.json`` files. ``--baseline NAME=FILE``
adds comparison methods from a JSON object ``{"<cell label>": calls}``;
repeat it per method. Files are ``<prefix>-calls``, ``<prefix>-matrix`` and
``<prefix>-walls``::

    rgpycrumbs surrogate plt-campaign results/clean-3ad8e8eb2-baker -o figs/baker --baseline "ASE ML-NEB=baselines/mlneb.json"
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
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@7b773c1321af0070919cbbfe8452269e165b540e",
# ]
# ///

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from chemparseplot.parse.surrogate.gpr_optim import parse_gpr_optim_campaign
from chemparseplot.plot.provenance import file_sha256, save_with_provenance
from chemparseplot.plot.surrogate import (
    plot_campaign_calls,
    plot_campaign_matrix,
    plot_campaign_walls,
    set_font,
)

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs surrogate plt-campaign")


def _baseline(_ctx, _param, values):
    out = {}
    for item in values:
        name, sep, path = item.partition("=")
        if not sep or not Path(path).is_file():
            raise click.BadParameter(
                f"{item!r}: expected NAME=FILE with an existing JSON file"
            )
        out[name] = Path(path)
    return out


@click.command()
@click.argument("record", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "prefix",
    required=True,
    type=click.Path(path_type=Path),
    help="Output prefix; the panel name and format are appended.",
)
@click.option(
    "--baseline",
    "baselines",
    multiple=True,
    callback=_baseline,
    help="NAME=FILE, JSON {label: calls}. Repeatable.",
)
@click.option(
    "--calls",
    type=click.Choice(["search_calls", "total_calls"]),
    default="search_calls",
    show_default=True,
    help="Which call count the dumbbell shows.",
)
@click.option("--linear", is_flag=True, help="Linear instead of log call axis.")
@click.option("--name", default=None, help="Legend name of this campaign.")
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
def main(*, record, prefix, baselines, calls, linear, name, fmt, dpi, font, font_dirs):
    """Draw campaign figures for RECORD."""
    set_font(font, font_dirs)
    table = parse_gpr_optim_campaign(record, name=name)
    if not table.cells:
        raise click.ClickException(f"{record} holds no <set>/<cell>/result.json")
    for method, path in baselines.items():
        table.baselines[method] = {
            k: float(v) for k, v in json.loads(path.read_text()).items()
        }
    hashes = {
        str(p.relative_to(record)): file_sha256(p)
        for p in sorted(record.glob("*/*/result.json"))
    }
    hashes.update({f"baseline:{m}": file_sha256(p) for m, p in baselines.items()})
    command = " ".join(["rgpycrumbs", "surrogate", "plt-campaign", *sys.argv[1:]])
    figs = {
        "calls": plot_campaign_calls(table, calls=calls, log=not linear),
        "matrix": plot_campaign_matrix(table),
        "walls": plot_campaign_walls(table),
    }
    for panel, fig in figs.items():
        out = prefix.with_name(f"{prefix.name}-{panel}.{fmt}")
        save_with_provenance(fig, out, {}, command=command, hashes=hashes, dpi=dpi)
        click.echo(out)


if __name__ == "__main__":
    main()
