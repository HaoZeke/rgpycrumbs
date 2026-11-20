"""Measure the periodic Euclidean force residual of a thermal instanton orbit.

The input is a full period sampled at uniform imaginary times. A force provider
evaluates those same geometries independently of the instanton optimization.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "numpy", "readcon", "scipy"]
# ///

import importlib
import importlib.util
import json
from pathlib import Path

import click
import numpy as np


def euclidean_force_residual(positions, forces, masses, temperature, ring_polymer=False):
    """Return F + M d²x/dτ² in eV/Å, with x in Å, masses in u, and T in K.

    Columns are mobile Cartesian degrees of freedom. Rows are uniformly
    spaced through a full thermal period, excluding the repeated endpoint.
    """
    constants = importlib.import_module("scipy.constants")

    positions, forces, masses = (
        np.asarray(a, dtype=float) for a in (positions, forces, masses)
    )
    if (
        positions.ndim != 2
        or positions.shape != forces.shape
        or len(positions) < 3
        or masses.shape != (positions.shape[1],)
        or not masses.size
    ):
        raise ValueError(
            "positions, forces, and masses must span the same mobile degrees of freedom"
        )
    if (
        not all(np.isfinite(a).all() for a in (positions, forces, masses))
        or not np.isfinite(temperature)
        or temperature <= 0
        or np.any(masses <= 0)
    ):
        raise ValueError("expected finite inputs and positive masses and temperature")
    period = constants.hbar / (constants.Boltzmann * temperature)
    omega = 2 * np.pi * np.fft.rfftfreq(len(positions), d=period / len(positions))
    if ring_polymer:
        step = period / len(positions)
        omega = 2 * np.sin(0.5 * omega * step) / step
    accel = np.fft.irfft(
        -(omega[:, None] ** 2) * np.fft.rfft(positions, axis=0), n=len(positions), axis=0
    )
    # kg Å/s² to eV/Å introduces two angstrom-to-metre factors.
    return (
        forces
        + (constants.atomic_mass * 1e-20 / constants.electron_volt) * masses * accel
    )


def orbit_extent(positions, masses):
    """Measure the Cartesian diameter in the stored frame and mass metric."""
    from scipy.spatial.distance import pdist

    xyz = np.asarray(positions, dtype=float)
    mass = np.asarray(masses, dtype=float)
    if xyz.ndim != 3 or len(xyz) < 2 or xyz.shape[1:] != (len(mass), 3):
        raise ValueError("expected at least two molecular geometries and one mass per atom")
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(mass) & (mass > 0)):
        raise ValueError("expected finite Cartesian positions and positive finite masses")
    unique = np.unique(xyz.reshape(len(xyz), -1), axis=0)
    if len(unique) == 1:
        atom_diameter, weighted_diameter = 0.0, 0.0
    else:
        xyz = unique.reshape(-1, len(mass), 3)
        atom_diameter = max(pdist(points).max() for points in np.moveaxis(xyz, 1, 0))
        weighted = xyz * np.sqrt(mass)[:, None]
        weighted_diameter = pdist(weighted.reshape(len(xyz), -1)).max()
    return {
        "distinct_geometries": len(unique),
        "max_atom_excursion_angstrom": float(atom_diameter),
        "mass_weighted_diameter_angstrom_sqrt_u": float(weighted_diameter),
    }


def _factory(spec):
    module_name, separator, name = spec.rpartition(":")
    if not separator or not name:
        raise click.BadParameter("expected module:function or /path/to/file.py:function")
    if module_name.endswith(".py"):
        location = importlib.util.spec_from_file_location(
            "orbit_force_provider", module_name
        )
        if location is None or location.loader is None:
            raise click.BadParameter(f"cannot load {module_name}")
        module = importlib.util.module_from_spec(location)
        location.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, name)


@click.command()
@click.argument("band", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--provider",
    required=True,
    help="Force provider factory as module:function or file.py:function.",
)
@click.option(
    "--options",
    default="{}",
    help="JSON keyword arguments for the force provider factory.",
)
@click.option(
    "--temperature",
    type=float,
    help="Temperature in K when absent from the frame metadata.",
)
@click.option(
    "--tolerance",
    type=click.FloatRange(min=0),
    help="Threshold for the largest residual in eV/Å.",
)
@click.option(
    "--ipi-half-ring",
    is_flag=True,
    help="Reflect a stored i-PI half-ring and use the finite-bead spring force.",
)
@click.option(
    "--atom-pair", type=(int, int), multiple=True,
    help="Report the minimum and maximum separation in Å for two readcon atom IDs.",
)
@click.option(
    "--compare-discretizations", is_flag=True,
    help="Report the Fourier and nearest-neighbour spring residuals using the same physical forces.",
)
def main(band, *, provider, options, temperature, tolerance, ipi_half_ring, atom_pair, compare_discretizations):
    """Report stationarity and independent validation costs for a readcon BAND.

    Each frame supplies tau_over_beta, atom IDs, masses and fixed components.
    Coordinates must be continuous across the period. The factory receives
    the band path and returns an object whose cartesian_forces(frame) gives
    an (atoms, 3) array in eV/Å. Its optional result_counts() returns the
    provider's cumulative computed and served geometry counts.
    """
    readcon = importlib.import_module("readcon")

    frames = list(readcon.iter_con(str(band)))
    if ipi_half_ring:
        frames.sort(key=lambda f: int(f.metadata["neb_bead"]))
        for i, frame in enumerate(frames):
            ledger = frame.metadata.get("evaluation_counts", {})
            if (
                frame.metadata["neb_bead"] != i
                or ledger.get("mode") != "rate"
                or ledger.get("stored_beads") != len(frames)
                or len(frames) < 2
            ):
                raise click.UsageError(
                    "expected the ordered beads of one i-PI rate half-ring"
                )
    else:
        if len(frames) < 3:
            raise click.UsageError("expected at least three beads spanning a full period")
        frames.sort(key=lambda f: float(f.metadata["tau_over_beta"]))
        phase = np.array([f.metadata["tau_over_beta"] for f in frames], dtype=float)
        if not (
            np.all((0 <= phase) & (phase < 1))
            and np.allclose(
                np.diff(np.r_[phase, phase[0] + 1]), 1 / len(frames), atol=1e-12, rtol=0
            )
        ):
            raise click.UsageError(
                "tau_over_beta must cover a full, evenly spaced period"
            )
    templ = frames[0]
    if temperature is None:
        temperature = templ.metadata.get("temperature_K")
    if temperature is None or any(
        not np.isclose(
            f.metadata.get("temperature_K", temperature), temperature, rtol=1e-12, atol=0
        )
        for f in frames
    ):
        raise click.UsageError("supply a temperature consistent with the orbit metadata")

    def identity(frame):
        return [(a.atom_id, a.symbol, a.mass, tuple(a.fixed)) for a in frame.atoms]

    atoms = identity(templ)
    if any(
        identity(f) != atoms or f.cell != templ.cell or f.angles != templ.angles
        for f in frames
    ):
        raise click.UsageError(
            "atom identities, masses, fixed masks and cell must agree across beads"
        )
    mobile = ~np.array([a.fixed for a in templ.atoms], dtype=bool)
    xyz = np.stack([f.xyz for f in frames])
    if not np.all(xyz[:, ~mobile] == xyz[0, ~mobile]):
        raise click.UsageError("fixed coordinates vary along the orbit")
    masses = np.broadcast_to(
        np.array([a.mass for a in templ.atoms])[:, None], mobile.shape
    )[mobile]
    pair_distances = []
    atom_ids = [a.atom_id for a in templ.atoms]
    for first, second in atom_pair:
        if first == second or atom_ids.count(first) != 1 or atom_ids.count(second) != 1:
            raise click.UsageError("each pair requires two distinct, unique atom IDs in the band")
        distances = np.linalg.norm(xyz[:, atom_ids.index(first)] - xyz[:, atom_ids.index(second)], axis=1)
        pair_distances.append({"atom_ids": [first, second], "min_angstrom": float(distances.min()),
                               "max_angstrom": float(distances.max())})
    extent = orbit_extent(xyz, [a.mass for a in templ.atoms])
    evaluator = _factory(provider)(band, **json.loads(options))
    read_counts = getattr(evaluator, "result_counts", lambda: None)
    before = read_counts()
    forces = np.stack([evaluator.cartesian_forces(f) for f in frames])
    if forces.shape != xyz.shape:
        raise ValueError("provider must return one Cartesian force vector per atom")
    if ipi_half_ring:
        xyz = np.concatenate((xyz, xyz[::-1]))
        forces = np.concatenate((forces, forces[::-1]))
    residual = euclidean_force_residual(
        xyz[:, mobile], forces[:, mobile], masses, temperature, ring_polymer=ipi_half_ring
    )
    after = read_counts()
    counts = None
    if before is not None and after is not None:
        counts = {k: after[k] - before[k] for k in ("computed", "served")}
    largest = float(np.max(np.abs(residual)))
    result = {
        "temperature_K": temperature,
        "beads": len(xyz),
        "discretization": "ring_polymer" if ipi_half_ring else "spectral",
        "mobile_coordinates": int(mobile.sum()),
        "geometry_extent": extent,
        "pair_distances": pair_distances,
        "max_residual_eV_per_angstrom": largest,
        "rms_residual_eV_per_angstrom": float(np.sqrt(np.mean(residual**2))),
        "evaluation_counts": {
            "scope": "validation",
            "geometry_requests": len(frames),
            "provider": getattr(evaluator, "provider_name", provider),
            "provider_results": counts,
        },
    }
    if compare_discretizations:
        other = euclidean_force_residual(xyz[:, mobile], forces[:, mobile], masses, temperature,
                                         ring_polymer=not ipi_half_ring)
        result["discretization_comparison"] = {
            result["discretization"]: {"max_eV_per_angstrom": largest,
                                       "rms_eV_per_angstrom": result["rms_residual_eV_per_angstrom"]},
            "spectral" if ipi_half_ring else "ring_polymer": {
                "max_eV_per_angstrom": float(np.max(np.abs(other))),
                "rms_eV_per_angstrom": float(np.sqrt(np.mean(other**2))),
            },
        }
    if tolerance is not None:
        result.update(
            tolerance_eV_per_angstrom=tolerance, within_tolerance=largest <= tolerance
        )
    click.echo(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
