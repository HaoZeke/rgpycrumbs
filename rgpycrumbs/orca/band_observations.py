"""Recover measured Cartesian gradients on a readcon band from native ORCA launches."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "numpy", "readcon"]
# ///

import hashlib
import json
import re
import tarfile
from io import StringIO
from pathlib import Path

import click
import numpy as np
import readcon
from ase.io.orca import read_geom_orcainp


def archive_records(path):
    records = {}
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            folder, _, leaf = member.name.rpartition("/")
            if leaf in {"input.inp", "output.engrad", "launch.json"}:
                group = records.setdefault(folder, {})
                if leaf in group:
                    raise ValueError(f"duplicate {member.name}")
                group[leaf] = archive.extractfile(member).read()
                if leaf == "input.inp":
                    group["timestamp"] = member.mtime
    if not records or any(len(group) != 4 for group in records.values()):
        raise ValueError(f"incomplete EnGrad archive {path}")
    return sorted(records.items(), key=lambda pair: pair[1]["timestamp"])


@click.command()
@click.argument("band", type=click.Path(exists=True, path_type=Path))
@click.argument("launches", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--native",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Native job summary for the primary archive.",
)
@click.option(
    "--inherited",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Archive of physical results available to the native job.",
)
@click.option("--hartree-ev", required=True, type=click.FloatRange(min=0, min_open=True))
@click.option(
    "--bohr-angstrom", required=True, type=click.FloatRange(min=0, min_open=True)
)
@click.option("--write-observations", required=True, type=click.Path(path_type=Path))
def main(
    band, launches, native, inherited, hartree_ev, bohr_angstrom, write_observations
):
    """Match every band image to independent input decks, EnGrad energies and gradients."""
    frames = readcon.read_con(str(band))
    if len(frames) < 3:
        raise click.UsageError("a band requires at least three images")
    symbols = [a.symbol for a in frames[0].atoms]
    identities = [(a.atom_id, a.symbol, a.mass) for a in frames[0].atoms]
    if any(
        [(a.atom_id, a.symbol, a.mass) for a in f.atoms] != identities for f in frames
    ):
        raise ValueError("atom identity differs across the band")
    xyz = np.asarray([f.xyz for f in frames]).reshape(len(frames), len(symbols), 3)
    energies = np.array([f.metadata["energy"] for f in frames])
    job = json.loads(native.read_bytes())
    out = job["evaluation_counts"]["rgpot_results"]["phases"]["output"]["computed"]
    matches = [[] for _ in frames]
    for index, archive in enumerate((launches, *inherited)):
        records = archive_records(archive)
        if not index and len(records) != job["physical_launches"]:
            raise ValueError("primary archive differs from native launch count")
        for k, (key, record) in enumerate(records):
            deck_text = re.sub(
                r"(?im)^\*\s+xyz\s+", "*xyz ", record["input.inp"].decode()
            )
            deck = read_geom_orcainp(StringIO(deck_text))
            if deck.get_chemical_symbols() != symbols:
                raise ValueError(f"atomic order mismatch in {key}")
            distances = np.max(abs(xyz - deck.positions), axis=(1, 2))
            if np.min(distances) > 5.1e-11:
                continue
            status = json.loads(record["launch.json"])
            if (
                status["returncode"]
                or not status["engrad_written"]
                or not status["normal_termination"]
                or status["scf_convergence_reports"] != 1
            ):
                raise ValueError(f"unsuccessful electronic observation in {key}")
            fields = [
                line.strip()
                for line in record["output.engrad"].splitlines()
                if line.strip() and not line.lstrip().startswith(b"#")
            ]
            n = int(fields[0])
            if n != len(symbols) or len(fields) != 2 + 4 * n:
                raise ValueError(f"invalid EnGrad shape in {key}")
            atomic = np.loadtxt(fields[-n:])
            energy = float(fields[1]) * hartree_ev
            if (
                not np.array_equal(atomic[:, 0], deck.numbers)
                or np.max(abs(atomic[:, 1:] * bohr_angstrom - deck.positions)) > 5e-8
            ):
                raise ValueError(
                    f"EnGrad identity or coordinates disagree with deck in {key}"
                )
            gradient = (
                np.array(fields[2 : 2 + 3 * n], dtype=float) * hartree_ev / bohr_angstrom
            ).reshape(n, 3)
            for i in np.flatnonzero(
                (distances <= 5.1e-11) & (abs(energies - energy) <= 1e-10)
            ):
                info = {
                    "archive_index": index,
                    "launch": key,
                    "output_phase": index == 0 and k >= len(records) - out,
                    "input_sha256": hashlib.sha256(record["input.inp"]).hexdigest(),
                    "engrad_sha256": hashlib.sha256(record["output.engrad"]).hexdigest(),
                    "energy_difference_eV": float(energy - energies[i]),
                    "deck_rounding_angstrom": float(distances[i]),
                }
                matches[i].append((info, energy, gradient))
    rows = []
    evidence = []
    for i, choices in enumerate(matches):
        if not choices:
            raise ValueError(f"missing verified native gradient for image {i}")
        chosen, energy, gradient = min(
            choices,
            key=lambda row: (
                not row[0]["output_phase"],
                abs(row[0]["energy_difference_eV"]),
                row[0]["deck_rounding_angstrom"],
            ),
        )
        rows.append(
            {
                "request": i + 1,
                "positions_angstrom": xyz[i].ravel().tolist(),
                "energy_eV": energy,
                "gradient_eV_per_angstrom": gradient.ravel().tolist(),
            }
        )
        evidence.append(
            {
                "image": i,
                "selected": chosen,
                "candidates": [rec[0] for rec in choices],
                "max_gradient_difference_eV_per_angstrom": max(
                    float(np.max(abs(g - gradient))) for _, _, g in choices
                ),
            }
        )
    write_observations.write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    click.echo(
        json.dumps(
            {
                "images": len(frames),
                "observations": evidence,
                "additional_surface_calls": 0,
                "hartree_eV": hartree_ev,
                "bohr_angstrom": bohr_angstrom,
                "input_sha256": [
                    {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (band, native, launches, *inherited)
                ],
                "output_sha256": hashlib.sha256(
                    write_observations.read_bytes()
                ).hexdigest(),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
