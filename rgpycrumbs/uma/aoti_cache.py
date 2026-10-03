#!/usr/bin/env python3
"""List, inspect and clear the UMA AOTI package cache.

.. versionadded:: 1.12.0

Reads sidecars and the metadata embedded in each ``.pt2``; needs no
torch. Prefer the dispatcher::

    rgpycrumbs uma aoti-cache list
    rgpycrumbs uma aoti-cache inspect omol-CHN-q0-s1
    rgpycrumbs uma aoti-cache clear --partial
    rgpycrumbs uma aoti-cache clear --all

``inspect`` exits 1 when a package's embedded metadata disagrees with
its key. ``clear`` skips an entry whose key is locked by a running
``prepare-aoti``.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click>=8.1",
# ]
# ///

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - script copy without the package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs uma aoti-cache")

from rgpycrumbs.uma._cache import (
    default_cache_dir,
    embedded_metadata,
    embedded_mismatches,
    entries,
    find_entry,
    remove,
    stale_partials,
)

_CACHE = click.option(
    "--cache",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Package cache [$RGPYCRUMBS_UMA_CACHE or $XDG_CACHE_HOME/rgpycrumbs/uma].",
)


def _root(cache: Path | None) -> Path:
    return cache if cache is not None else default_cache_dir()


@click.group()
def main() -> None:
    """Manage UMA AOTI packages prepared by ``rgpycrumbs uma prepare-aoti``."""


@main.command("list")
@_CACHE
@click.option("--json", "as_json", is_flag=True, help="One JSON object per entry.")
def list_cmd(cache: Path | None, as_json: bool) -> None:
    """One row per package: key fields, size and compile time."""
    root = _root(cache)
    found = entries(root)
    if as_json:
        for e in found:
            click.echo(
                json.dumps({"path": str(e.path), "size": e.size, **(e.sidecar or {})})
            )
        return
    click.echo(f"cache {root}: {len(found)} package(s)")
    for e in found:
        key = e.key
        if key is None:
            click.echo(f"  {e.path.name}  (no sidecar: not a cache hit)")
            continue
        secs = (e.sidecar or {}).get("compile_seconds")
        click.echo(
            f"  {e.stem}\n"
            f"      {key.formula} q={key.charge} s={key.spin} task={key.task} "
            f"model={key.model} dtype={key.dtype} box={key.molecular_box} "
            f"batch={key.batch_max}\n"
            f"      {key.device} [{key.target}] torch={key.torch} "
            f"fairchem={key.fairchem} {e.size / 2**20:.1f} MiB"
            + (f" compiled {secs:.1f} s" if isinstance(secs, (int, float)) else "")
        )


@main.command()
@_CACHE
@click.argument("name")
def inspect(cache: Path | None, name: str) -> None:
    """Sidecar and embedded metadata of NAME, checked against its key.

    NAME is a path, a file name, a stem, a unique stem prefix, or the
    digest at the end of the stem.
    """
    try:
        entry = find_entry(_root(cache), name)
    except LookupError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"package  {entry.path}")
    click.echo(f"size     {entry.size} bytes")
    try:
        meta = embedded_metadata(entry.path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"cannot read embedded metadata: {exc}") from exc
    click.echo("embedded " + json.dumps(meta, indent=2, sort_keys=True))
    if entry.sidecar is None:
        click.echo("sidecar  none: lookups miss this package")
        return
    click.echo("sidecar  " + json.dumps(entry.sidecar, indent=2, sort_keys=True))
    key = entry.key
    if key is None:
        raise click.ClickException("sidecar holds no valid key")
    bad = embedded_mismatches(meta, key)
    if bad:
        for line in bad:
            click.echo(f"MISMATCH {line}", err=True)
        raise SystemExit(1)
    click.echo("ok       embedded metadata matches the key")


@main.command()
@_CACHE
@click.argument("names", nargs=-1)
@click.option("--all", "everything", is_flag=True, help="Every package.")
@click.option("--partial", is_flag=True, help="Leftovers of interrupted compiles.")
@click.option("--dry-run", is_flag=True, help="Say what would go; delete nothing.")
def clear(
    cache: Path | None,
    names: tuple[str, ...],
    everything: bool,
    partial: bool,
    dry_run: bool,
) -> None:
    """Delete packages named by NAMES, or --all, and/or --partial leftovers."""
    root = _root(cache)
    if not (names or everything or partial):
        raise click.UsageError("name packages, or pass --all or --partial")
    try:
        targets = entries(root) if everything else [find_entry(root, n) for n in names]
    except LookupError as exc:
        raise click.ClickException(str(exc)) from exc
    busy = 0
    for entry in targets:
        if dry_run:
            click.echo(f"would remove {entry.path.name}")
        elif remove(root, entry):
            click.echo(f"removed {entry.path.name}")
        else:
            busy += 1
            click.echo(f"skipped {entry.path.name}: being prepared", err=True)
    if partial:
        for path in stale_partials(root):
            click.echo(f"{'would remove' if dry_run else 'removed'} {path.name}")
            if not dry_run:
                shutil.rmtree(path, ignore_errors=True)
    if busy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
