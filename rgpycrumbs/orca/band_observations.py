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


def measured_width(seed_path, identities, measurements, target, fraction):
    """Find the unique measured initial interior band between observed crossings."""
    frames = readcon.read_con(str(seed_path))
    if any(
        [(a.atom_id, a.symbol, a.mass) for a in f.atoms] != identities for f in frames
    ):
        raise ValueError("initial molecular identities differ from the output band")
    source = np.array([f.xyz for f in frames]).reshape(len(frames), -1)
    masses = np.repeat(np.sqrt([a[2] for a in identities]), 3)
    points = np.array([r[1] for r in measurements]).reshape(len(measurements), -1)
    energies = np.array([r[2] for r in measurements])
    turns = []
    for i, j in ((0, 1), (-1, -2)):
        v = source[j] - source[i]
        t = (points - source[i]) @ v / (v @ v)
        distance = np.max(abs(points - source[i] - t[:, None] * v), axis=1)
        edge = (distance <= 1e-9) & (t >= -1e-7) & (t < 1)
        if not np.any(edge & (energies > target)) or not np.any(
            edge & (energies < target)
        ):
            raise ValueError("no two-sided measured crossing on a terminal chord")
        k = int(np.argmin(np.where(edge, abs(energies - target), np.inf)))
        if abs(energies[k] - target) > 1e-6:
            raise ValueError("terminal measurements do not resolve the crossing")
        source[i] += t[k] * v
        turns.append(
            {
                "launch": measurements[k][0],
                "chord_fraction": float(t[k]),
                "energy_minus_target_eV": float(energies[k] - target),
            }
        )
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(source, axis=0) * masses, axis=1))]
    uniform = np.stack(
        [np.interp(np.linspace(0, arc[-1], len(frames)), arc, c) for c in source.T],
        axis=1,
    )
    indices = []
    offsets = []
    for image in uniform[1:-1]:
        delta = np.max(abs(points - image), axis=1)
        found = np.flatnonzero(delta <= 2e-9)
        if not len(found):
            raise ValueError(
                "no measured interior image on the reconstructed initial arc"
            )
        indices.append(found)
        offsets.append(delta)
    blocks = [
        int(i)
        for i in indices[0]
        if all(i + j in candidates for j, candidates in enumerate(indices))
    ]
    if len(blocks) != 1:
        raise ValueError(
            "the physical archive does not identify one initial interior sequence"
        )
    block = np.arange(blocks[0], blocks[0] + len(indices))
    heights = energies[block] - target
    if np.min(heights) <= 0:
        raise ValueError("initial interior image lies outside the turning surfaces")
    return {
        "seed_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        "fraction": fraction,
        "absolute_width_eV": float(fraction * np.max(heights)),
        "crossings": turns,
        "measured_interior_launches": [measurements[k][0] for k in block],
        "interior_energies_eV": energies[block].tolist(),
        "max_reconstructed_interior_coordinate_error_angstrom": float(
            max(d[k] for d, k in zip(offsets, block, strict=True))
        ),
    }


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
@click.option(
    "--initial-band",
    type=click.Path(exists=True, path_type=Path),
    help="Two-sided molecular seed with its original interior images.",
)
@click.option(
    "--width-fraction",
    type=click.FloatRange(min=0, min_open=True),
    help="Relative momentum regularization used by the native optimizer.",
)
def main(
    band,
    launches,
    native,
    inherited,
    hartree_ev,
    bohr_angstrom,
    write_observations,
    initial_band,
    width_fraction,
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
    if (initial_band is None) != (width_fraction is None):
        raise click.UsageError("give both --initial-band and --width-fraction")
    primary = []
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
            if np.min(distances) > 5.1e-11 and not (initial_band and index == 0):
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
            if initial_band and index == 0:
                primary.append((key, deck.positions, energy))
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
                "atoms": [
                    {"id": i, "symbol": symbol, "mass_amu": mass}
                    for i, symbol, mass in identities
                ],
                "observations": evidence,
                **(
                    {
                        "momentum_width": measured_width(
                            initial_band,
                            identities,
                            primary,
                            job["lineb"]["energy"],
                            width_fraction,
                        )
                    }
                    if initial_band
                    else {}
                ),
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
