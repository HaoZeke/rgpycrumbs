"""Relate a converged i-PI half-ring to its recorded physical energy gradients."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "ipi>=3", "numpy", "readcon", "scipy"]
# ///

import hashlib
import json
import re
from pathlib import Path

import click
import numpy as np
import readcon
from ase.io import read
from ipi.utils.units import unit_to_user

from rgpycrumbs.eon.orbit_residual import euclidean_force_residual


def coordinate_key(x):
    return np.asarray(x, dtype="<f8").reshape(-1).tobytes()


def final_tables(directory, symbols):
    files = []
    for path in directory.glob("*instanton_FINAL_*.xyz"):
        match = re.fullmatch(r"(.*instanton_FINAL_)(\d+)\.xyz", path.name)
        if match:
            files.append((int(match[2]), match[1], path))
    if not files or len({prefix for _, prefix, _ in files}) != 1:
        raise ValueError("expected the coordinate output of one native instanton run")
    step, prefix, position_file = max(files)
    force_file = directory / f"{prefix}forces_{step}.xyz"
    tables = []
    for path, unit in [(position_file, "angstrom"), (force_file, "atomic_unit")]:
        frames = read(path, index=":", format="xyz")
        headers = path.read_text().splitlines()[1::len(symbols) + 2]
        if (len(headers) != len(frames) or
                any(f"positions{{{unit}}}" not in header for header in headers) or
                any(list(frame.symbols) != symbols for frame in frames)):
            raise ValueError(f"unexpected native units or atom order in {path}")
        tables.append(np.asarray([frame.positions for frame in frames]))
    factor = unit_to_user("energy", "electronvolt", 1.) / unit_to_user("length", "angstrom", 1.)
    return step, tables[0], tables[1] * factor


@click.command()
@click.argument("band", type=click.Path(exists=True, path_type=Path))
@click.argument("ledger", type=click.Path(exists=True, path_type=Path))
@click.option("--write-observations", type=click.Path(path_type=Path), help="Write the matched solver JSONL rows in bead order.")
@click.option("--native-directory", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Compare with the native FINAL coordinate and atomic-unit force tables.")
def main(band, ledger, write_observations, native_directory):
    """Check an i-PI readcon BAND against its original geometry-request LEDGER."""
    if write_observations and write_observations.resolve() in (band.resolve(), ledger.resolve()):
        raise click.UsageError("the observation output must differ from both inputs")
    frames = list(readcon.read_con(str(band)))
    if len(frames) < 2:
        raise ValueError("expected at least two half-ring geometries")
    first = frames[0]
    counts = first.metadata["evaluation_counts"]
    temp = first.metadata["temperature_K"]
    symbols = [atom.symbol for atom in first.atoms]
    identity = [(a.atom_id, a.symbol, a.mass, tuple(a.fixed)) for a in first.atoms]
    sources = counts.get("input_seed", {})
    if counts.get("mode") != "rate" or not counts.get("optimizer_converged") or counts["stored_beads"] != len(frames):
        raise ValueError("the input must be one converged native i-PI rate half-ring")
    for i, frame in enumerate(frames):
        if (frame.metadata["neb_bead"] != i or frame.metadata["temperature_K"] != temp or
                frame.metadata["evaluation_counts"] != counts or
                [(a.atom_id, a.symbol, a.mass, tuple(a.fixed)) for a in frame.atoms] != identity):
            raise ValueError("bead indices, temperature, counts or atom identity differ across the orbit")
    xyz = np.asarray([f.xyz for f in frames])
    keys = [coordinate_key(pos) for pos in xyz]
    wanted = set(keys)
    selected = {}
    digest = hashlib.sha256()
    from ase.data import atomic_numbers
    numbers = [atomic_numbers[s] for s in symbols]
    with ledger.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            key = coordinate_key(row["positions_angstrom"])
            if key not in wanted or key in selected:
                continue
            if row.get("atomic_numbers", numbers) != numbers or (
                sources.get("surface_sha256") and row.get("surface_sha256") != sources["surface_sha256"]
            ):
                raise ValueError("matched observation has different surface or atom order")
            selected[key] = row
    if wanted != selected.keys():
        raise ValueError("the ledger lacks exact Cartesian coordinates for some orbit beads")
    rows = [selected[key] for key in keys]
    gradient = np.asarray([r["gradient_eV_per_angstrom"] for r in rows]).reshape(xyz.shape)
    masses = np.repeat([a.mass for a in first.atoms], 3)
    mobile = np.array([a.fixed for a in first.atoms]).reshape(-1) == 0
    full_xyz = np.concatenate([xyz, xyz[::-1]]).reshape(2*len(frames), -1)
    full_forces = -np.concatenate([gradient, gradient[::-1]]).reshape(2*len(frames), -1)
    report = {"temperature_K": temp, "beads": len(full_xyz), "additional_surface_requests": 0,
              "orbit_sha256": hashlib.sha256(band.read_bytes()).hexdigest(), "ledger_sha256": digest.hexdigest(),
              "actual_solver_rows_for_final_geometry": [r["request"] for r in rows]}
    for label, discrete in [("ring_polymer", True), ("spectral", False)]:
        r = euclidean_force_residual(full_xyz[:, mobile], full_forces[:, mobile], masses[mobile], temp, ring_polymer=discrete)
        report[label] = {"max_eV_per_angstrom": float(abs(r).max()), "rms_eV_per_angstrom": float(np.sqrt(np.mean(r*r)))}
    if native_directory:
        step, pos, force = final_tables(native_directory, symbols)
        if pos.shape != xyz.shape or force.shape != xyz.shape:
            raise ValueError("native FINAL tables and readcon half-ring must have the same dimensions")
        err_x, err_f = float(abs(pos-xyz).max()), float(abs(force+gradient).max())
        if not (err_x < 1e-9 and err_f < 1e-6):
            raise ValueError(f"native FINAL data disagree by {err_x} angstrom and {err_f} eV/angstrom")
        report.update(source_step=step, max_native_coordinate_difference_angstrom=err_x,
                      max_difference_from_native_force_ledger_eV_per_angstrom=err_f)
    if write_observations:
        with write_observations.open("w") as stream:
            for row in rows:
                print(json.dumps(row, allow_nan=False), file=stream)
    click.echo(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
