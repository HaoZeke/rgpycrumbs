"""Match native ORCA EnGrad observations to both ends of a readcon band."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "numpy", "readcon"]
# ///

import hashlib
import json
from pathlib import Path

import click
import numpy as np
import readcon
from ase.data import atomic_numbers
from ase.units import Bohr


@click.command()
@click.argument("band", type=click.Path(exists=True, path_type=Path))
@click.argument("launches", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--target-ev", type=float, help="Tunneling energy in eV.")
@click.option("--level-record", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Read the tunneling energy from one frozen LI level.")
@click.option("--hartree-ev", required=True, type=float, help="Conversion factor used by the native potential.")
@click.option("--require-bracket", is_flag=True, help="Require both observations to be below the tunneling energy.")
def main(band, launches, target_ev, level_record, hartree_ev, require_bracket):
    """Find the two physical endpoint observations and report E(endpoint) minus E(target)."""
    if (level_record is None) == (target_ev is None):
        raise click.UsageError("give either --level-record or --target-ev")
    if level_record is not None:
        target_ev = json.loads(level_record.read_text())["energy"]
    frames = readcon.read_con(str(band))
    if len(frames) < 2 or not np.isfinite([target_ev, hartree_ev]).all() or hartree_ev <= 0:
        raise click.ClickException("expected two distinct frames, finite energy and a positive conversion factor")
    endpoints = [frames[0], frames[-1]]
    matches = {}
    for file in sorted(launches.glob("*/output.engrad"), key=lambda f: f.stat().st_mtime_ns):
        with file.open() as stream:
            lines = [line for line in stream if line.strip() and not line.lstrip().startswith("#")]
        n = int(lines[0])
        if len(lines) != 2 + 4 * n:
            raise click.ClickException(f"incomplete EnGrad file {file}")
        geometry = np.loadtxt(lines[-n:], ndmin=2)
        energy = float(lines[1]) * hartree_ev
        for i, frame in enumerate(endpoints):
            numbers = [atomic_numbers[a.symbol] for a in frame.atoms]
            if i in matches or not np.array_equal(numbers, geometry[:, 0]):
                continue
            # ORCA prints seven decimal places in its EnGrad coordinate block.
            error = np.max(np.abs(geometry[:, 1:] * Bohr - np.asarray(frame.xyz)))
            if error > 5e-8:
                continue
            matches[i] = {"frame": [0, len(frames) - 1][i], "launch": file.parent.name,
                          "engrad_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                          "energy_eV": energy, "energy_minus_target_eV": energy - target_ev,
                          "max_coordinate_rounding_angstrom": float(error)}
        if len(matches) == 2:
            break
    if len(matches) != 2:
        raise click.ClickException("missing measured energy for a seed endpoint")
    below = all(v["energy_minus_target_eV"] < 0 for v in matches.values())
    click.echo(json.dumps({"source_con_sha256": hashlib.sha256(band.read_bytes()).hexdigest(),
                           "target_eV": target_ev, "hartree_eV": hartree_ev, "both_below_target": below,
                           "endpoints": [matches[i] for i in range(2)]}, indent=2, allow_nan=False))
    if require_bracket and not below:
        raise click.ClickException("endpoint energies do not bracket an interior orbit")


if __name__ == "__main__":
    main()
