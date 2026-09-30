#!/usr/bin/env python3
"""Centroid and ring-polymer spread from a PIMD bead trajectory.

.. versionadded:: 1.12.0

A path-integral molecular dynamics run carries P replicas (beads) of every
atom. This collapses them to one con frame per sampled step: the positions
are the centroid over the beads, and the readcon ``spreads`` section holds
each atom's root-mean-square displacement from that centroid along x, y and
z in Angstrom.

Inputs, either of which may be CPMD ``TRAJECTORY`` text (step index, x, y, z,
vx, vy, vz per atom, positions in bohr, velocities ignored) or extended xyz
(positions in Angstrom); the format is read from the first line:

- several files, one per replica, with the same number of frames and atoms;
- one file whose frames hold P x N atoms, replica-major, with
  ``--replicas P``.

The trajectory carries no cell, masses or fixed flags. ``--reference`` (a
con file, first frame) supplies symbols, masses, cell and fixed flags;
otherwise ``--symbols`` (a whitespace-separated list, or xyz symbols) and
``--cell`` supply what the frames need.

``--time-average`` averages the centroid over the sampled frames and writes
one frame. Its spread combines the imaginary-time spread (rms over frames of
the bead spread) and the thermal spread (rms of the centroid about its
average) in quadrature, which equals the rms of all beads in all frames about
the time-averaged centroid. Both parts are in the frame metadata as
``spread_imaginary_time_rms`` and ``spread_thermal_rms``, rms over atoms and
components in Angstrom.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "numpy",
#   "readcon>=0.16.0",
# ]
# ///

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np

log = logging.getLogger(__name__)

#: Bohr radius in Angstrom.
BOHR_TO_ANGSTROM = 0.529177210544


def _is_xyz(path: Path) -> bool:
    with open(path) as fh:
        for line in fh:
            tokens = line.split()
            if tokens:
                return len(tokens) == 1 and tokens[0].isdigit()
    msg = f"{path} is empty"
    raise ValueError(msg)


def read_xyz(path: Path) -> tuple[np.ndarray, list[str]]:
    """Frames ``(n_frames, n_atoms, 3)`` in Angstrom and the symbols."""
    frames, symbols = [], None
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].split()[0])
        except ValueError as exc:
            msg = f"{path}: line {i + 1} is not an atom count"
            raise ValueError(msg) from exc
        block = lines[i + 2 : i + 2 + n]
        if len(block) != n:
            msg = f"{path}: frame {len(frames)} is truncated"
            raise ValueError(msg)
        rows = [ln.split() for ln in block]
        syms = [r[0] for r in rows]
        if symbols is None:
            symbols = syms
        elif syms != symbols:
            msg = f"{path}: frame {len(frames)} changes the species order"
            raise ValueError(msg)
        frames.append(np.array([[float(v) for v in r[1:4]] for r in rows]))
        i += 2 + n
    if not frames:
        msg = f"{path} holds no frames"
        raise ValueError(msg)
    if len({f.shape for f in frames}) != 1:
        msg = f"{path}: frames differ in atom count"
        raise ValueError(msg)
    return np.stack(frames), symbols


def read_cpmd_trajectory(path: Path) -> np.ndarray:
    """Frames ``(n_frames, n_atoms, 3)`` in Angstrom from a ``TRAJECTORY``.

    A frame is a run of consecutive rows with one step index.
    """
    data = np.loadtxt(path, ndmin=2)
    if data.size == 0 or data.shape[1] < 4:
        msg = f"{path} has no step, x, y, z columns"
        raise ValueError(msg)
    steps = data[:, 0]
    starts = np.flatnonzero(np.r_[True, steps[1:] != steps[:-1]])
    sizes = np.diff(np.r_[starts, len(steps)])
    if len(set(sizes.tolist())) != 1:
        msg = f"{path}: frames differ in atom count {sorted(set(sizes.tolist()))}"
        raise ValueError(msg)
    pos = data[:, 1:4] * BOHR_TO_ANGSTROM
    return pos.reshape(len(starts), int(sizes[0]), 3)


def read_trajectory(path: Path) -> tuple[np.ndarray, list[str] | None]:
    """Positions ``(n_frames, n_atoms, 3)`` in Angstrom, plus xyz symbols."""
    path = Path(path)
    if _is_xyz(path):
        return read_xyz(path)
    return read_cpmd_trajectory(path), None


def load_replicas(
    paths: list[Path], replicas: int | None = None
) -> tuple[np.ndarray, list[str] | None]:
    """Bead positions ``(P, n_frames, n_atoms, 3)`` in Angstrom and symbols.

    One path splits its frames replica-major into *replicas* beads; several
    paths are one replica each.
    """
    if not paths:
        msg = "no trajectory file given"
        raise ValueError(msg)
    if len(paths) == 1:
        frames, symbols = read_trajectory(paths[0])
        p = 1 if replicas is None else replicas
        if p < 1 or frames.shape[1] % p:
            msg = (
                f"{paths[0]}: {frames.shape[1]} atoms per frame do not split "
                f"into {p} replicas"
            )
            raise ValueError(msg)
        n = frames.shape[1] // p
        if symbols is not None:
            ring = [symbols[k * n : (k + 1) * n] for k in range(p)]
            if any(r != ring[0] for r in ring):
                msg = f"{paths[0]}: replicas differ in species order"
                raise ValueError(msg)
            symbols = ring[0]
        beads = frames.reshape(frames.shape[0], p, n, 3).transpose(1, 0, 2, 3)
        return beads, symbols
    if replicas is not None and replicas != len(paths):
        msg = f"--replicas {replicas} disagrees with {len(paths)} files"
        raise ValueError(msg)
    loaded = [read_trajectory(p) for p in paths]
    shapes = [f.shape for f, _ in loaded]
    if len(set(shapes)) != 1:
        detail = "; ".join(
            f"{p}: {s[0]} frames x {s[1]} atoms"
            for p, s in zip(paths, shapes, strict=True)
        )
        msg = f"replicas differ in frame or atom count ({detail})"
        raise ValueError(msg)
    symbols = loaded[0][1]
    if any(s is not None and s != symbols for _, s in loaded):
        msg = "replicas differ in species order"
        raise ValueError(msg)
    return np.stack([f for f, _ in loaded]), symbols


def centroid_spread(beads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and per-component rms spread, both ``(n_frames, n_atoms, 3)``."""
    centroid = beads.mean(axis=0)
    spread = np.sqrt(((beads - centroid) ** 2).mean(axis=0))
    return centroid, spread


def time_average(
    centroid: np.ndarray, spread: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average over frames.

    Returns the mean centroid, the total spread, and its imaginary-time and
    thermal parts, each ``(n_atoms, 3)``.
    """
    mean = centroid.mean(axis=0)
    imaginary = np.sqrt((spread**2).mean(axis=0))
    thermal = np.sqrt(((centroid - mean) ** 2).mean(axis=0))
    return mean, np.sqrt(imaginary**2 + thermal**2), imaginary, thermal


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt((a**2).mean()))


def _template(reference, symbols, cell, n_atoms):
    """Per-atom symbol, mass, fixed flags plus cell and angles."""
    if reference is not None:
        from readcon import read_con  # noqa: PLC0415

        frames = read_con(str(reference))
        if not frames:
            msg = f"{reference} holds no frames"
            raise ValueError(msg)
        ref = frames[0]
        if len(ref.atoms) != n_atoms:
            msg = f"{reference} has {len(ref.atoms)} atoms, trajectory has {n_atoms}"
            raise ValueError(msg)
        atoms = [(a.symbol, a.mass, a.fixed) for a in ref.atoms]
        return atoms, list(ref.cell), list(ref.angles)
    if symbols is None:
        msg = "no species: give --reference, --symbols or an xyz trajectory"
        raise ValueError(msg)
    if len(symbols) != n_atoms:
        msg = f"{len(symbols)} symbols for {n_atoms} atoms"
        raise ValueError(msg)
    if cell is None:
        msg = "no cell: give --reference or --cell"
        raise ValueError(msg)
    return [(s, None, None) for s in symbols], list(cell), [90.0, 90.0, 90.0]


def build_frames(
    positions: np.ndarray,
    spreads: np.ndarray,
    metadata: list[dict],
    *,
    reference=None,
    symbols=None,
    cell=None,
):
    """readcon frames, one per row of *positions* ``(F, N, 3)``."""
    import readcon  # noqa: PLC0415

    template, box, angles = _template(reference, symbols, cell, positions.shape[1])
    frames = []
    for pos, spr, md in zip(positions, spreads, metadata, strict=True):
        atoms = [
            readcon.Atom(
                symbol=sym,
                x=float(pos[i, 0]),
                y=float(pos[i, 1]),
                z=float(pos[i, 2]),
                fixed=fixed,
                atom_id=i,
                mass=mass,
                sx=float(spr[i, 0]),
                sy=float(spr[i, 1]),
                sz=float(spr[i, 2]),
            )
            for i, (sym, mass, fixed) in enumerate(template)
        ]
        frames.append(readcon.ConFrame(cell=box, angles=angles, atoms=atoms, metadata=md))
    return frames


def collapse(
    beads: np.ndarray, *, every: int = 1, average: bool = False
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Positions, spreads and per-frame metadata from bead positions."""
    if every < 1:
        msg = "--every must be at least 1"
        raise ValueError(msg)
    p = beads.shape[0]
    index = np.arange(beads.shape[1])[::every]
    centroid, spread = centroid_spread(beads[:, ::every])
    if not average:
        md = [
            {
                "replicas": p,
                "frame_index": int(k),
                "spread_max": float(s.max()),
            }
            for k, s in zip(index, spread, strict=True)
        ]
        return centroid, spread, md
    mean, total, imaginary, thermal = time_average(centroid, spread)
    md = {
        "replicas": p,
        "frame_index": 0,
        "frames_averaged": len(index),
        "spread_max": float(total.max()),
        "spread_imaginary_time_rms": _rms(imaginary),
        "spread_thermal_rms": _rms(thermal),
    }
    return mean[None], total[None], [md]


@click.command()
@click.argument("trajectories", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option("--replicas", type=click.IntRange(min=1), help="Replicas in one file.")
@click.option("--every", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--reference", type=click.Path(exists=True, path_type=Path))
@click.option("--symbols", "symbols_file", type=click.Path(exists=True, path_type=Path))
@click.option("--cell", nargs=3, type=float, help="Box lengths a b c in Angstrom.")
@click.option("--time-average", "average", is_flag=True, help="One averaged frame.")
@click.option(
    "--out",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("centroid.con"),
    show_default=True,
)
def main(  # noqa: PLR0917
    trajectories, replicas, every, reference, symbols_file, cell, average, out
):
    """Write the bead centroid and spread of TRAJECTORIES as a con file."""
    logging.basicConfig(level="INFO", format="%(message)s")
    try:
        for t in trajectories:
            if not t.is_file():
                msg = f"{t} does not exist"
                raise FileNotFoundError(msg)
        beads, xyz_symbols = load_replicas(list(trajectories), replicas)
        positions, spreads, md = collapse(beads, every=every, average=average)
        symbols = Path(symbols_file).read_text().split() if symbols_file else xyz_symbols
        frames = build_frames(
            positions,
            spreads,
            md,
            reference=reference,
            symbols=symbols,
            cell=cell or None,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    import readcon  # noqa: PLC0415

    readcon.write_con(str(out), frames)
    log.info(
        "%d frame(s) of %d atoms from %d replicas to %s; largest spread %.4f A",
        len(frames),
        positions.shape[1],
        beads.shape[0],
        out,
        max(m["spread_max"] for m in md),
    )


if __name__ == "__main__":
    main()
