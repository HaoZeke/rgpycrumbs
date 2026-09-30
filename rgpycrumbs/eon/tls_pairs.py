#!/usr/bin/env python3
"""Two-level-system pairs from an eOn aKMC run, with tunnelling splittings.

.. versionadded:: 1.11.0

A two-level system (TLS) in a glass is a pair of adjacent minima that the
system tunnels between at about one kelvin. aKMC already holds the pairs:
each row of a state's ``processtable`` is a reactant, a saddle and a product,
kept in ``procdata/``. For every pair this writes the asymmetry, the barrier,
the mass-weighted distance between the minima, and the one-dimensional WKB
tunnelling splitting with the TLS energy ``sqrt(delta**2 + delta0**2)``.

Where eOn has run an NEB on the pair, the splitting comes from that band: eOn
writes it into the first frame of ``neb.con`` (``tunnel_splitting``,
``hbar_omega_reactant``, ...), computed along the mass-weighted band. Without
a band, the pair gets a three-point estimate through reactant, saddle and
product, labelled ``three_point``: a screen, not a result.

Quantities carry units through pint; the CSV states them in its header.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "numpy",
#   "pint>=0.24",
#   "rich",
#   "ase",
#   "readcon>=0.14.9",
# ]
# ///

from __future__ import annotations

import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pint
from readcon import read_con
from rich.logging import RichHandler

log = logging.getLogger("rich")

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

#: hbar in the band's native units, eV**0.5 amu**0.5 Angstrom.
HBAR = Q_(1, "hbar").to("(eV * dalton)**0.5 * angstrom").magnitude
#: Boltzmann constant in eV / K.
KB = Q_(1, "boltzmann_constant").to("eV / K").magnitude
#: Displacement past which an atom counts as moved, Angstrom.
MOVED_CUTOFF = 0.1

#: CSV columns, units in the names.
FIELDS = (
    "state",
    "process",
    "delta_eV",
    "barrier_eV",
    "distance_mw_amu^0.5_A",
    "max_displacement_A",
    "moved_atoms",
    "delta0_eV",
    "tls_energy_eV",
    "tls_K",
    "path",
)

#: The processtable columns eOn writes (eon/akmcstate.py).
_SADDLE_ENERGY, _PRODUCT_ENERGY, _BARRIER = 1, 4, 6


@dataclass
class Pair:
    """One pair of minima and what it would be as a two-level system."""

    state: str
    process: int
    delta: pint.Quantity
    barrier: pint.Quantity
    distance_mw: pint.Quantity
    max_displacement: pint.Quantity
    moved_atoms: int
    delta0: pint.Quantity
    tls_energy: pint.Quantity
    path: str

    @property
    def tls_kelvin(self) -> pint.Quantity:
        return (self.tls_energy / Q_(KB, "eV/K")).to("K")

    def row(self) -> dict[str, object]:
        return {
            "state": self.state,
            "process": self.process,
            "delta_eV": self.delta.to("eV").magnitude,
            "barrier_eV": self.barrier.to("eV").magnitude,
            "distance_mw_amu^0.5_A": self.distance_mw.to(
                "dalton**0.5 * angstrom"
            ).magnitude,
            "max_displacement_A": self.max_displacement.to("angstrom").magnitude,
            "moved_atoms": self.moved_atoms,
            "delta0_eV": self.delta0.to("eV").magnitude,
            "tls_energy_eV": self.tls_energy.to("eV").magnitude,
            "tls_K": self.tls_kelvin.magnitude,
            "path": self.path,
        }


def _atoms(con: Path):
    return read_con(str(con))[0].to_ase()


def _displacement(a, b) -> np.ndarray:
    """Minimum-image displacement from ``a`` to ``b`` under the cell of ``a``."""
    dr = b.get_positions() - a.get_positions()
    cell = np.asarray(a.get_cell())
    if not np.any(cell):
        return dr
    frac = dr @ np.linalg.inv(cell)
    frac = np.where(a.get_pbc(), frac - np.round(frac), frac)
    return frac @ cell


def _mass_weighted(dr: np.ndarray, masses: np.ndarray) -> float:
    if np.any(masses <= 0):
        msg = "every atom needs a positive mass for a mass-weighted distance"
        raise ValueError(msg)
    return float(np.sqrt(np.sum(masses[:, None] * dr**2)))


def three_point_splitting(d1: float, d2: float, barrier: float, delta: float) -> float:
    """WKB splitting of a double well through reactant, saddle and product.

    Each side is a cubic flat at both ends (``V (3 t**2 - 2 t**3)``), whose
    curvature at a minimum is ``6 V / d**2``. The level is the higher of the
    two harmonic ground states, the prefactor frequency the geometric mean,
    the same convention eOn's client uses along a band. Energies in eV,
    distances in amu**0.5 Angstrom.
    """
    if d1 <= 0 or d2 <= 0 or barrier <= max(0.0, delta):
        msg = "the saddle has to lie between and above both minima"
        raise ValueError(msg)
    hw1 = HBAR * math.sqrt(6.0 * barrier / d1**2)
    hw2 = HBAR * math.sqrt(6.0 * (barrier - delta) / d2**2)
    level = max(0.5 * hw1, delta + 0.5 * hw2)
    t = np.linspace(0.0, 1.0, 2001)
    smooth = 3 * t**2 - 2 * t**3
    s = np.concatenate([t * d1, d1 + t[1:] * d2])
    v = np.concatenate([barrier * smooth, barrier + (delta - barrier) * smooth[1:]])
    gap = np.sqrt(np.clip(2.0 * (v - level), 0.0, None))
    action = float(np.trapezoid(gap, s)) / HBAR
    return math.sqrt(hw1 * hw2) / math.pi * math.exp(-action)


def _band_values(neb_con: Path) -> dict[str, float] | None:
    """The tunnelling values eOn wrote on a band's first frame, if any."""
    md = read_con(str(neb_con))[0].metadata
    return md if "tunnel_splitting" in md else None


def pairs_of_state(state_dir: Path, neb_root: Path | None) -> list[Pair]:
    table = state_dir / "processtable"
    proc = state_dir / "procdata"
    out: list[Pair] = []
    for line in table.read_text().splitlines()[1:]:
        cols = line.split()
        if len(cols) < 9:
            continue
        pid = int(cols[0])
        e_saddle = float(cols[_SADDLE_ENERGY])
        barrier = float(cols[_BARRIER])
        e_reactant = e_saddle - barrier
        delta = float(cols[_PRODUCT_ENERGY]) - e_reactant
        try:
            reactant = _atoms(proc / f"reactant_{pid}.con")
            saddle = _atoms(proc / f"saddle_{pid}.con")
            product = _atoms(proc / f"product_{pid}.con")
        except (OSError, IndexError) as exc:
            log.warning("%s process %d: %s", state_dir.name, pid, exc)
            continue
        masses = reactant.get_masses()
        dr = _displacement(reactant, product)
        norms = np.linalg.norm(dr, axis=1)
        band = neb_root / state_dir.name / str(pid) / "neb.con" if neb_root else None
        values = _band_values(band) if band and band.is_file() else None
        if values is not None:
            delta0 = values["tunnel_splitting"]
            path = "band"
        else:
            try:
                delta0 = three_point_splitting(
                    _mass_weighted(_displacement(reactant, saddle), masses),
                    _mass_weighted(_displacement(saddle, product), masses),
                    barrier,
                    delta,
                )
            except ValueError as exc:
                log.warning("%s process %d: %s", state_dir.name, pid, exc)
                continue
            path = "three_point"
        out.append(
            Pair(
                state=state_dir.name,
                process=pid,
                delta=Q_(delta, "eV"),
                barrier=Q_(barrier, "eV"),
                distance_mw=Q_(_mass_weighted(dr, masses), "dalton**0.5 * angstrom"),
                max_displacement=Q_(float(norms.max()), "angstrom"),
                moved_atoms=int(np.count_nonzero(norms > MOVED_CUTOFF)),
                delta0=Q_(delta0, "eV"),
                tls_energy=Q_(math.hypot(delta, delta0), "eV"),
                path=path,
            )
        )
    return out


def _state_dirs(akmc_dir: Path) -> list[Path]:
    states = akmc_dir / "states" if (akmc_dir / "states").is_dir() else akmc_dir
    if (states / "processtable").is_file():
        return [states]
    dirs = [p for p in states.iterdir() if (p / "processtable").is_file()]
    return sorted(dirs, key=lambda p: (len(p.name), p.name))


@click.command()
@click.argument("akmc_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), help="CSV (default stdout)."
)
@click.option(
    "--neb-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="NEB jobs laid out as NEB_ROOT/<state>/<process>/neb.con; their "
    "band splittings replace the three-point estimate.",
)
@click.option("--max-asymmetry", type=float, help="Keep |delta| <= this, eV.")
@click.option("--max-barrier", type=float, help="Keep barriers <= this, eV.")
@click.option("--max-kelvin", type=float, help="Keep TLS energies <= this, K.")
def main(akmc_dir, *, output, neb_root, max_asymmetry, max_barrier, max_kelvin):
    """Write the TLS pairs an aKMC run found, with WKB tunnelling splittings."""
    logging.basicConfig(level="INFO", handlers=[RichHandler()], format="%(message)s")
    pairs = [p for s in _state_dirs(akmc_dir) for p in pairs_of_state(s, neb_root)]
    kept = [
        p
        for p in pairs
        if (max_asymmetry is None or abs(p.delta.magnitude) <= max_asymmetry)
        and (max_barrier is None or p.barrier.magnitude <= max_barrier)
        and (max_kelvin is None or p.tls_kelvin.magnitude <= max_kelvin)
    ]
    rows = [p.row() for p in kept]
    fields = list(FIELDS)
    handle = open(output, "w", newline="") if output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output:
            handle.close()
    log.info("%d of %d pairs kept", len(kept), len(pairs))


if __name__ == "__main__":
    main()
