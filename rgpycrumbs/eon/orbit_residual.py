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


def euclidean_force_residual(positions, forces, masses, temperature):
    """Return F + M d²x/dτ² in eV/Å, with x in Å, masses in u, and T in K.

    Columns are mobile Cartesian degrees of freedom. Rows are uniformly
    spaced through a full thermal period, excluding the repeated endpoint.
    """
    from scipy.constants import Boltzmann, atomic_mass, electron_volt, hbar

    positions, forces, masses = (np.asarray(a, dtype=float) for a in (positions, forces, masses))
    if (positions.ndim != 2 or positions.shape != forces.shape or len(positions) < 3
            or masses.shape != (positions.shape[1],) or not masses.size):
        raise ValueError("positions, forces, and masses must span the same mobile degrees of freedom")
    if (not all(np.isfinite(a).all() for a in (positions, forces, masses))
            or not np.isfinite(temperature) or temperature <= 0 or np.any(masses <= 0)):
        raise ValueError("expected finite inputs and positive masses and temperature")
    period = hbar / (Boltzmann * temperature)
    omega = 2 * np.pi * np.fft.rfftfreq(len(positions), d=period / len(positions))
    accel = np.fft.irfft(-omega[:, None] ** 2 * np.fft.rfft(positions, axis=0),
                         n=len(positions), axis=0)
    # kg Å/s² to eV/Å introduces two angstrom-to-metre factors.
    return forces + (atomic_mass * 1e-20 / electron_volt) * masses * accel


def _factory(spec):
    module_name, separator, name = spec.rpartition(":")
    if not separator or not name:
        raise click.BadParameter("expected module:function or /path/to/file.py:function")
    if module_name.endswith(".py"):
        location = importlib.util.spec_from_file_location("orbit_force_provider", module_name)
        if location is None or location.loader is None:
            raise click.BadParameter(f"cannot load {module_name}")
        module = importlib.util.module_from_spec(location)
        location.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, name)


@click.command()
@click.argument("band", type=click.Path(exists=True, path_type=Path))
@click.option("--provider", required=True, help="Force provider factory as module:function or file.py:function.")
@click.option("--options", default="{}", help="JSON keyword arguments for the force provider factory.")
@click.option("--temperature", type=float, help="Temperature in K when absent from the frame metadata.")
@click.option("--tolerance", type=click.FloatRange(min=0), help="Threshold for the largest residual in eV/Å.")
def main(band, provider, options, temperature, tolerance):
    """Report stationarity and independent validation costs for a readcon BAND.

    Each frame supplies tau_over_beta, atom IDs, masses and fixed components.
    Coordinates must be continuous across the period. The factory receives
    the band path and returns an object whose cartesian_forces(frame) gives
    an (atoms, 3) array in eV/Å. Its optional result_counts() returns the
    provider's cumulative computed and served geometry counts.
    """
    import readcon

    frames = list(readcon.iter_con(str(band)))
    if len(frames) < 3:
        raise click.UsageError("expected at least three beads spanning a full period")
    frames.sort(key=lambda f: float(f.metadata["tau_over_beta"]))
    phase = np.array([f.metadata["tau_over_beta"] for f in frames], dtype=float)
    if not (np.all((0 <= phase) & (phase < 1))
            and np.allclose(np.diff(np.r_[phase, phase[0] + 1]), 1 / len(frames), atol=1e-12, rtol=0)):
        raise click.UsageError("tau_over_beta must cover a full, evenly spaced period")
    templ = frames[0]
    if temperature is None:
        temperature = templ.metadata.get("temperature_K")
    if temperature is None or any(
        not np.isclose(f.metadata.get("temperature_K", temperature), temperature, rtol=1e-12, atol=0)
        for f in frames
    ):
        raise click.UsageError("supply a temperature consistent with the orbit metadata")
    def identity(frame):
        return [(a.atom_id, a.symbol, a.mass, tuple(a.fixed)) for a in frame.atoms]

    atoms = identity(templ)
    if any(identity(f) != atoms or f.cell != templ.cell or f.angles != templ.angles for f in frames):
        raise click.UsageError("atom identities, masses, fixed masks and cell must agree across beads")
    mobile = ~np.array([a.fixed for a in templ.atoms], dtype=bool)
    xyz = np.stack([f.xyz for f in frames])
    if not np.all(xyz[:, ~mobile] == xyz[0, ~mobile]):
        raise click.UsageError("fixed coordinates vary along the orbit")
    masses = np.broadcast_to(np.array([a.mass for a in templ.atoms])[:, None], mobile.shape)[mobile]
    evaluator = _factory(provider)(band, **json.loads(options))
    read_counts = getattr(evaluator, "result_counts", lambda: None)
    before = read_counts()
    forces = np.stack([evaluator.cartesian_forces(f) for f in frames])
    if forces.shape != xyz.shape:
        raise ValueError("provider must return one Cartesian force vector per atom")
    residual = euclidean_force_residual(xyz[:, mobile], forces[:, mobile], masses, temperature)
    after = read_counts()
    counts = None
    if before is not None and after is not None:
        counts = {k: after[k] - before[k] for k in ("computed", "served")}
    largest = float(np.max(np.abs(residual)))
    result = {"temperature_K": temperature, "beads": len(frames), "mobile_coordinates": int(mobile.sum()),
              "max_residual_eV_per_angstrom": largest,
              "rms_residual_eV_per_angstrom": float(np.sqrt(np.mean(residual**2))),
              "evaluation_counts": {"scope": "validation", "geometry_requests": len(frames),
                                    "provider": getattr(evaluator, "provider_name", provider),
                                    "provider_results": counts}}
    if tolerance is not None:
        result.update(tolerance_eV_per_angstrom=tolerance, within_tolerance=largest <= tolerance)
    click.echo(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
