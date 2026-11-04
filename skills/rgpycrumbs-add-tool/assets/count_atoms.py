#!/usr/bin/env python3
"""Count the atoms of each element in the first frame of an XYZ file.

Prefer the dispatcher::

    rgpycrumbs geom count-atoms molecule.xyz
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click>=8.1",
# ]
# ///

from __future__ import annotations

from collections import Counter
from pathlib import Path

import click


def count_elements(path: Path) -> Counter[str]:
    """Element symbol to atom count, for the first frame of *path*."""
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0])
    return Counter(line.split()[0] for line in lines[2 : 2 + natoms])


@click.command()
@click.argument("xyz", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(xyz: Path) -> None:
    """Print one line per element of XYZ: the symbol and the count."""
    for symbol, count in sorted(count_elements(xyz).items()):
        click.echo(f"{symbol} {count}")


if __name__ == "__main__":
    main()
