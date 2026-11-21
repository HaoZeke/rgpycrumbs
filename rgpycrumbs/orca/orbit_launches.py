"""Identify physical ORCA launches for orbit beads and displaced image Hessians."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "numpy", "readcon", "scipy"]
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
from scipy.spatial import cKDTree


def read_decks(archive, symbols):
    decks, reports = {}, {}
    with tarfile.open(archive, "r|*") as files:
        for member in files:
            if not member.isfile():
                continue
            name, leaf = member.name.rsplit("/", 1)
            if leaf == "input.inp":
                if name in decks:
                    raise ValueError(f"duplicate input deck in {name}")
                decks[name] = member.mtime, files.extractfile(member).read().decode()
            elif leaf == "launch.json":
                reports[name] = json.load(files.extractfile(member))
    if not decks or set(decks) != set(reports):
        raise ValueError("expected a matching status record for each input deck")
    members = sorted(decks, key=lambda name: decks[name][0])
    if len({decks[name][0] for name in members}) != len(members):
        raise ValueError(
            "launch archive timestamps cannot resolve physical execution order"
        )
    coordinates, names = [], []
    for name in members:
        text = decks[name][1]
        geometry = read_geom_orcainp(
            StringIO(re.sub(r"(?im)^\*\s+xyz\s+", "*xyz ", text))
        )
        if geometry.get_chemical_symbols() != symbols:
            raise ValueError(f"atom identity differs in {name}")
        launch = reports[name]
        if not (
            launch["returncode"] == 0
            and launch["normal_termination"]
            and launch["engrad_written"]
            and launch["scf_convergence_reports"] == 1
        ):
            raise ValueError(f"expected one successful EnGrad and SCF in {name}")
        names.append(name)
        coordinates.append(geometry.positions.ravel())
    return names, np.asarray(coordinates)


@click.command()
@click.argument("band", type=click.Path(exists=True, path_type=Path))
@click.argument("orbit", type=click.Path(exists=True, path_type=Path))
@click.argument("native", type=click.Path(exists=True, path_type=Path))
@click.argument("launches", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--inherited-launches",
    type=click.Path(exists=True, path_type=Path),
    help="ORCA launch archive for the measurement that populated the inherited cache.",
)
@click.option(
    "--deck-decimals", type=click.IntRange(1, 16), default=10, show_default=True
)
def main(band, orbit, native, launches, inherited_launches, deck_decimals):
    """Audit an EnGrad archive with a full Cartesian finite-difference stencil."""
    summary = json.loads(native.read_text())
    frames = list(readcon.iter_con(str(orbit)))
    images = list(readcon.iter_con(str(band)))
    signature = [(a.atom_id, a.symbol, a.mass) for a in images[0].atoms]
    if any(
        [(a.atom_id, a.symbol, a.mass) for a in f.atoms] != signature
        for f in images + frames
    ):
        raise ValueError("band and orbit atom identities differ")
    centers = np.array([f.xyz for f in images]).reshape(len(images), -1)
    ring = np.array([f.xyz for f in frames]).reshape(len(frames), -1)
    n = len(ring)
    if n < 4 or n % 2 or n != summary["lineb"]["beads"]:
        raise ValueError("expected the full even-bead native orbit")
    step = summary["lineb"]["image_hessian_step_angstrom"]
    if not np.isfinite(step) or step <= 0:
        raise ValueError("expected a positive finite Cartesian displacement")
    names, decks = read_decks(launches, [s[1] for s in signature])
    if len(names) != summary["physical_launches"]:
        raise ValueError("launch count differs from the native job")
    # The deck prints fixed decimal coordinates. The cache uses unrounded geometry.
    tol = 0.5 * 10**-deck_decimals + np.finfo(float).eps * max(1.0, abs(decks).max()) * 4
    offset = step * np.eye(centers.shape[1])
    stencil = (
        centers[:, None, None, :] + np.stack((offset, -offset), axis=1)[None]
    ).reshape(-1, centers.shape[1])
    distances, indices = cKDTree(stencil).query(decks, p=np.inf)
    located = np.flatnonzero(distances <= tol)
    if (
        len(located) != len(stencil)
        or len(set(indices[located])) != len(stencil)
        or np.any(np.diff(located) != 1)
    ):
        raise ValueError(
            "archive does not contain a contiguous complete image Hessian stencil"
        )
    prefix, start = int(located[0]), int(located[-1] + 1)
    nout = summary["evaluation_counts"]["rgpot_results"]["phases"]["output"]["computed"]
    stop = len(decks) - nout
    phases = (
        ["pre_rate"] * prefix
        + ["image_hessian"] * len(located)
        + ["orbit"] * (stop - start)
        + ["output"] * nout
    )
    if start > stop or len(phases) != len(decks):
        raise ValueError("native output boundary lies inside the Hessian stencil")
    rows = [{"launch": name, "phase": ph} for name, ph in zip(names, phases, strict=True)]
    for i in located:
        center, component, side = np.unravel_index(
            indices[i], (len(centers), centers.shape[1], 2)
        )
        rows[i].update(image=int(center), coordinate=int(component), sign=(1, -1)[side])
    # readcon fixed-decimal coordinates can round distinct callback inputs together.
    first = {}
    for i, row in enumerate(ring):
        first.setdefault(row.tobytes(), i)
    available = decks[:start]
    inherited_count = 0
    if inherited_launches:
        if summary.get("cache_state") not in {"seeded", "calibrated"} or not summary.get(
            "inherited_cache"
        ):
            raise ValueError(
                "an inherited archive requires a native cache inheritance record"
            )
        inherited_names, inherited_decks = read_decks(
            inherited_launches, [s[1] for s in signature]
        )
        inherited_count = len(inherited_names)
        available = np.concatenate((available, inherited_decks))
    k = start
    preexisting = []
    for i, bead in enumerate(ring):
        if k < stop and (err := abs(decks[k] - bead).max()) <= tol:
            rows[k].update(bead=i, max_deck_rounding_angstrom=float(err))
            k += 1
        else:
            if np.any(np.max(abs(decks[start:k] - bead), axis=1) <= tol):
                continue
            if not np.any(np.max(abs(available - bead), axis=1) <= tol):
                raise ValueError(f"missing launch and prior deck for bead {i}")
            if first[bead.tobytes()] == i:
                preexisting.append(i)
    if k != stop:
        raise ValueError("central orbit deck order differs from the observed ring")
    half = n // 2
    duplicates = [r for r in rows[start:stop] if r["bead"] > half]
    mirror = np.max(np.abs(ring[1:half] - ring[:half:-1]), axis=1)
    if np.any(mirror > tol):
        raise ValueError(
            "return branch differs from its outgoing partner at deck resolution"
        )
    if len({r.tobytes() for r in ring[: half + 1]}) != half + 1:
        raise ValueError("outgoing half has repeated geometries")
    for i in range(stop, len(rows)):
        distance = np.max(abs(centers - decks[i]), axis=1)
        if np.count_nonzero(distance <= tol) != 1:
            raise ValueError("output launch cannot be uniquely matched to a band image")
        rows[i]["image"] = int(np.argmin(distance))
    physical = stop - start
    result = {
        "measured": {
            "pre_rate_launches": prefix,
            "image_displacement_launches": len(located),
            "central_orbit_requests": n,
            "central_orbit_launches": physical,
            "central_orbit_served": n - physical,
            "output_launches": nout,
            "job_launches": len(rows),
            "distinct_binary64_ring_geometries": len(first),
            "distinct_outgoing_including_turns": half + 1,
            "already_cached_ring_first_indices": preexisting,
            "return_only_launches": len(duplicates),
            "max_mirror_coordinate_difference_angstrom": float(mirror.max()),
        },
        "fixed_forward_reflection_prediction": {
            "central_orbit_launches": physical - len(duplicates),
            "central_orbit_served": n - physical + len(duplicates),
            "job_launches": len(rows) - len(duplicates),
            "assumption": "all other requests and cached results fixed, exact reuse of reversed forward coordinates",
        },
        "input_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (band, orbit, native, launches)
        },
        "additional_surface_requests": 0,
        "physical_launches_in_order": rows,
    }
    if inherited_launches:
        result["inherited_cache"] = {
            "archive_sha256": hashlib.sha256(inherited_launches.read_bytes()).hexdigest(),
            "archive_launches": inherited_count,
            "native_inheritance": summary["inherited_cache"],
        }
    click.echo(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
