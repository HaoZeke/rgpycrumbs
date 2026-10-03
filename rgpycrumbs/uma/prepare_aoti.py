#!/usr/bin/env python3
"""Look up or compile the UMA AOTI package rgpot's UmaPot loads for one system.

.. versionadded:: 1.12.0

The key is the exact composition, charge, spin multiplicity, task head,
model, dtype, molecular box, band size, device and its ISA or compute
capability, and the torch and fairchem versions. A hit costs one stat
and one sidecar read; torch is imported once per environment, to probe
its versions and target, and the answer is remembered. A miss runs rgpot
``scripts/export_uma_aoti.py`` in this script's environment, checks the
package's embedded metadata against the key, and installs it atomically
under a per-key lock, so concurrent runs for one system compile once.

Prefer the dispatcher::

    rgpycrumbs uma prepare-aoti reactant.con --molecular-box 25
    rgpycrumbs uma prepare-aoti radical.xyz --charge 0 --spin 2 --device cuda
    rgpycrumbs uma prepare-aoti reactant.con --dry-run

The package path is the last line on stdout. ``RGPOT_EXPORT_UMA`` or
``--exporter`` names the exporter; ``rgpycrumbs uma aoti-cache`` lists,
inspects and clears the cache.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click>=8.1",
#   "numpy>=1.26",
#   "ase>=3.26",
#   "readcon>=0.14.5",
#   "fairchem-core>=2.22",
#   "vesin>=0.6",
# ]
# ///

from __future__ import annotations

import json
from pathlib import Path

import click

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs uma prepare-aoti")

from rgpycrumbs.uma._io import atomic_numbers_of, charge_spin_of, load_atoms
from rgpycrumbs.uma._prepare import prepare_uma_aoti


@click.command()
@click.argument("atoms", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--charge", type=int, default=None, help="Total charge [atoms.info or 0].")
@click.option(
    "--spin",
    type=int,
    default=None,
    help="Spin multiplicity 2S+1 [atoms.info, else the minimum for the electron parity].",
)
@click.option("--task", default="omol", show_default=True, help="UMA task head.")
@click.option("--model", default="uma-s-1p1", show_default=True, help="UMA checkpoint.")
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="cpu, cuda or cuda:N; must match the UmaPot device.",
)
@click.option(
    "--molecular-box",
    type=float,
    default=0.0,
    show_default=True,
    help="Trace in this cube (angstrom) with a complete intramolecular graph; 0 is off.",
)
@click.option(
    "--batch-max",
    type=int,
    default=0,
    show_default=True,
    help="Band package for this many same-composition images; 0 is single-system.",
)
@click.option(
    "--cache",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Package cache [$RGPYCRUMBS_UMA_CACHE or $XDG_CACHE_HOME/rgpycrumbs/uma].",
)
@click.option(
    "--exporter",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="rgpot scripts/export_uma_aoti.py [$RGPOT_EXPORT_UMA].",
)
@click.option(
    "--python",
    "python",
    default=None,
    help="Interpreter that runs the exporter [this script's].",
)
@click.option("--force", is_flag=True, help="Compile even when the cache has the key.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the key, hit or miss, and the exporter command; compile nothing.",
)
def main(
    atoms: Path,
    *,
    charge: int | None,
    spin: int | None,
    task: str,
    model: str,
    device: str,
    molecular_box: float,
    batch_max: int,
    cache: Path | None,
    exporter: Path | None,
    python: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Print the path of the UMA AOTI package for ATOMS, compiling it on a miss."""
    loaded = load_atoms(atoms)
    numbers = atomic_numbers_of(loaded)
    try:
        q, s = charge_spin_of(loaded, charge=charge, spin=spin)
        result = prepare_uma_aoti(
            numbers,
            charge=q,
            spin=s,
            task=task,
            model=model,
            device=device,
            molecular_box=molecular_box,
            batch_max=batch_max,
            cache_dir=cache,
            atoms_path=atoms,
            exporter=exporter,
            python=python,
            force=force,
            dry_run=dry_run,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    key = result.key
    click.echo(f"key {key.stem()}", err=True)
    if dry_run:
        click.echo(json.dumps(key.as_dict(), indent=2, sort_keys=True), err=True)
        click.echo("hit" if result.hit else "miss", err=True)
        if result.command:
            click.echo("would run: " + " ".join(result.command), err=True)
    elif result.hit:
        click.echo("hit", err=True)
    else:
        click.echo(f"compiled in {result.compile_seconds:.1f} s", err=True)
    click.echo(str(result.path))


if __name__ == "__main__":
    main()
