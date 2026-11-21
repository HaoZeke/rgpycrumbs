"""Normalize a molecular instanton flux using a finite-bead harmonic reactant."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "numpy", "readcon", "scipy"]
# ///

import json
from pathlib import Path

import click
import numpy as np
from scipy import constants


def log_inertia_det(positions, masses):
    """Inertia of a molecule or full ring, with each replica weighted by 1/P."""
    positions = np.asarray(positions, dtype=float).reshape(-1, len(masses), 3)
    center = np.average(positions.mean(axis=0), axis=0, weights=masses)
    centered = positions - center
    second = np.einsum("pni,n,pnj->ij", centered, masses, centered) / len(positions)
    moments = np.linalg.eigvalsh(np.trace(second) * np.eye(3) - second)
    if not np.all(np.isfinite(moments) & (moments > 0)):
        raise ValueError("expected a nonlinear molecule with positive moments of inertia")
    return np.log(moments).sum()


def molecular_rate(level, minimum, atoms, ring):
    """Return a single-channel rate in s⁻¹ for an unconstrained nonlinear molecule."""
    rate = level["rate"]
    negative_modes = rate.get("negative_modes")
    if negative_modes is not None and negative_modes != 1:
        raise ValueError(f"instanton must have one retained unstable mode, found {negative_modes}")
    p = rate["beads"]
    temperature = rate["temperature"]
    masses = atoms.get_masses()
    freq = np.asarray(minimum["signed_wavenumbers_cm-1"], dtype=float)
    if (p < 3 or ring.size != p * 3 * len(masses) or rate["modes"] != p * 3 * len(masses) - 7
            or not np.isfinite(ring).all() or not np.all(np.isfinite(masses) & (masses > 0))):
        raise ValueError("expected a full molecular ring with six rigid modes and one path mode removed")
    if freq.shape != (3 * len(masses) - 6,) or not np.all(np.isfinite(freq) & (freq > 0)):
        raise ValueError("expected positive internal reactant wavenumbers")
    if not np.isclose(level["reference_energy"], minimum["electronic_energy_eV"], atol=1e-7, rtol=0):
        raise ValueError("reactant and flux must use the same electronic energy reference")
    controls = level["controls"]
    time_s = 1e-10 * np.sqrt(constants.atomic_mass / constants.electron_volt)
    beta = 1 / (controls["boltzmann"] * temperature)
    if not np.isfinite(beta) or beta <= 0 or controls["hbar"] <= 0:
        raise ValueError("expected positive thermal energy and Planck constant")
    # Each internal oscillator contributes [2 sinh(P asinh(beta hbar omega / 2P))]^-1.
    omega = 2 * np.pi * constants.c * 100 * freq * time_s
    arg = p * np.arcsinh(beta * controls["hbar"] * omega / (2 * p))
    log_q_vib = -np.sum(arg + np.log(-np.expm1(-2 * arg)))
    # For each rigid direction the non-centroid spring frequencies yield
    # product_{k=1}^{P-1} [2 sin(pi k/P)]^-1 = 1/P.
    log_q_rigid_springs = -6 * np.log(p)
    log_rotation = 0.5 * (log_inertia_det(ring, masses) - log_inertia_det(atoms.positions, masses))
    log_flux = rate["log_k_times_Zr"] + beta * minimum["electronic_energy_eV"]
    log_rate = log_flux - log_q_vib - log_q_rigid_springs + log_rotation - np.log(time_s)
    if not np.isfinite(log_rate):
        raise ValueError("nonfinite normalized rate")
    return dict(temperature_K=temperature, beads=p, log10_rate_per_s=log_rate / np.log(10),
                log_flux_reactant_electronic_zero=log_flux, log_Q_vib=log_q_vib,
                log_Q_rigid_springs=log_q_rigid_springs, log_rotation_ratio=log_rotation,
                time_unit_s=time_s, path_multiplicity=1)


def export_observed_ring(path, level, atoms, ring, observations, end, inputs, run_counts=None):
    """Write the last measured rate gradients of a Rust LI level as a readcon orbit."""
    from collections import deque
    import hashlib
    from itertools import islice
    import readcon

    phases = level["evaluation_counts"]
    p = level["rate"]["beads"]
    total = phases["surface_calls"]["total"]
    if phases["surface_calls"]["rate"] < p or end < total:
        raise ValueError("expected the measured rate evaluations at the end of the energy level")
    with observations.open() as stream:
        records = deque(islice(stream, end), maxlen=p)
    if len(records) != p:
        raise ValueError("incomplete molecular rate observation ledger")
    rows = list(map(json.loads, records))
    xyz = np.asarray([row["positions_angstrom"] for row in rows], dtype="<f8").reshape(p,-1)
    if xyz.tobytes() != np.asarray(ring, dtype="<f8").reshape(p,-1).tobytes():
        raise ValueError("observation coordinates differ from the native equal-time beads")
    gradients = np.asarray([r["gradient_eV_per_angstrom"] for r in rows]).reshape(p,len(atoms),3)
    energies = np.array([r["energy_eV"] for r in rows])
    if not np.isfinite(gradients).all() or not np.isfinite(energies).all():
        raise ValueError("expected measured finite energy and gradient at every bead")
    def digest(source):
        with source.open("rb") as f:
            return hashlib.file_digest(f,"sha256").hexdigest()
    provenance = {"producer": "rgsaddle", "representation": "exported measured rate orbit",
                  "source_sha256": {k:digest(v) for k,v in inputs.items()},
                  "reference_energy_eV": level["reference_energy"], "level": level["level"]}
    counts = dict(phases)
    if run_counts is not None:
        meta = json.loads(run_counts.read_text())
        if meta["reference_energy"] != level["reference_energy"]:
            raise ValueError("the provider counts refer to a different electronic reference")
        counts["provider_run"] = {"scope": "run", **meta["worker_counts"]}
        provenance["rgsaddle_revision"] = meta.get("rgsaddle_pin")
        provenance["source_sha256"]["run_counts"] = digest(run_counts)
    rate, controls = level["rate"], level["controls"]
    lineb = {"beads": p, "temperature": rate["temperature"], "half_period": rate["period"]/2,
             "molecular": True, "hbar": controls["hbar"], "boltzmann": controls["boltzmann"],
             "log_rate_over_reactant": rate["log_k_times_Zr"],
             "replica_clock": rate["replica_clock"], "hessian": rate["method"],
             "spectrum": {"modes": rate["modes"], "negative_modes": rate["negative_modes"]}}
    template = atoms.copy()
    def frames():
        for i in range(p):
            template.positions = xyz[i].reshape(-1,3)
            frame = readcon.ConFrame.from_ase(template)
            ids = [a.atom_id for a in frame.atoms]
            frame.metadata = {**frame.metadata, "frame_index": i, "tau_over_beta": i/p,
                "temperature_K": rate["temperature"], "lineb": lineb,
                "orbit_provenance": provenance, "evaluation_counts": counts,
                "lineb_orbit": {"mobile_atom_ids": ids, "energy_eV": float(energies[i]),
                                "gradients_eV_per_angstrom": gradients[i].tolist()}}
            yield frame
    if path.resolve() in {src.resolve() for src in inputs.values()}:
        raise ValueError("the exported ring must have its own output path")
    readcon.write_con(str(path), frames(), precision=17)


def eon_orbit_rate(path, minimum, atoms, confirm_same_energy_zero):
    """Normalize the measured full orbit written by an eOn LI-NEB molecular job."""
    import readcon

    if minimum.get("unstable_modes", 0) != 0:
        raise ValueError("the equilibrium spectrum must identify a minimum")
    frames = list(readcon.iter_con(str(path)))
    if len(frames) < 3:
        raise ValueError("expected at least three measured orbit frames")
    provenance = frames[0].metadata.get("orbit_provenance", {})
    reference = provenance.get("reference_energy_eV")
    if reference is None and not confirm_same_energy_zero:
        raise ValueError("confirm that the spectrum and orbit use the same surface and electronic zero with --confirm-same-energy-zero")
    lineb = frames[0].metadata.get("lineb", {})
    if not isinstance(lineb, dict) or lineb.get("molecular") is not True or lineb.get("beads") != len(frames):
        raise ValueError("expected a complete molecular eOn rate orbit")
    spectrum = lineb.get("spectrum")
    if not isinstance(spectrum, dict) or spectrum.get("negative_modes") != 1 or "modes" not in spectrum:
        raise ValueError("expected the measured fluctuation index and retained mode count")
    for name in ("temperature", "log_rate_over_reactant"):
        value = lineb.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"expected a finite {name} in eOn rate metadata")
    controls = {}
    for name in ("hbar", "boltzmann"):
        value = lineb.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"expected explicit positive {name} in eOn metadata")
        controls[name] = value
    signature = [(a.atom_id, a.symbol, a.mass) for a in frames[0].atoms]
    ids = [row[0] for row in signature]
    masses = np.array([row[2] for row in signature])
    if len(set(ids)) != len(ids) or not np.all(np.isfinite(masses) & (masses > 0)):
        raise ValueError("expected distinct readcon atom IDs and positive isotope masses")
    symbols = [row[1] for row in signature]
    if len(atoms) != len(ids) or atoms.constraints or np.any(atoms.pbc):
        raise ValueError("expected an unconstrained nonperiodic equilibrium molecule")
    identity = "species and isotope masses"
    id_array = next((name for name in ("atom_id", "id") if name in atoms.arrays), None)
    if id_array is not None:
        equilibrium_ids = list(atoms.arrays[id_array])
        if set(equilibrium_ids) != set(ids):
            raise ValueError("equilibrium atom IDs disagree with the orbit")
        atoms = atoms[[equilibrium_ids.index(i) for i in ids]]
        identity = "atom IDs, species and isotope masses"
    else:
        # The rotational inertia is invariant to permutations of identical isotopes.
        order = sorted(range(len(ids)), key=lambda i: (symbols[i], masses[i]))
        equilibrium = sorted(range(len(atoms)), key=lambda i: (atoms[i].symbol, atoms[i].mass))
        atoms = atoms[[equilibrium[order.index(i)] for i in range(len(ids))]]
    if atoms.get_chemical_symbols() != symbols or not np.allclose(atoms.get_masses(), masses, rtol=0, atol=1e-6):
        raise ValueError("equilibrium species or isotope masses disagree with the orbit")
    for i, frame in enumerate(frames):
        meta = frame.metadata
        if (meta.get("orbit_provenance", {}) != provenance or meta.get("lineb") != lineb or meta.get("temperature_K") != lineb.get("temperature")
                or not np.isclose(meta.get("tau_over_beta", np.nan), i / len(frames), rtol=0, atol=1e-12)):
            raise ValueError("inconsistent job metadata or out-of-order imaginary-time phase")
        units = meta.get("units", {})
        if units.get("length", "angstrom").lower() not in {"angstrom", "ångström"} or units.get("mass", "amu").lower() not in {"amu", "u", "dalton"}:
            raise ValueError("expected eOn Cartesian coordinates in angstrom and masses in atomic mass units")
        if frame.cell != frames[0].cell or frame.angles != frames[0].angles:
            raise ValueError("orbit frames must share one cell")
        observed = meta.get("lineb_orbit", {})
        mobile = observed.get("mobile_atom_ids", [])
        if ([(a.atom_id, a.symbol, a.mass) for a in frame.atoms] != signature
                or any(any(a.fixed) for a in frame.atoms) or sorted(mobile) != sorted(ids)):
            raise ValueError("expected consistent atom identities and a fully mobile molecular orbit")
    rate = dict(temperature=lineb["temperature"], beads=len(frames), modes=spectrum["modes"],
                negative_modes=spectrum["negative_modes"], log_k_times_Zr=lineb["log_rate_over_reactant"])
    level = dict(controls=controls, reference_energy=minimum["electronic_energy_eV"] if reference is None else reference, rate=rate)
    result = molecular_rate(level, minimum, atoms, np.stack([frame.xyz for frame in frames]))
    result["energy_reference"] = {"same_surface_and_zero_confirmed_by_caller": confirm_same_energy_zero,
                                  "independent_orbit_reference_energy_eV": reference,
                                  "reactant_electronic_energy_eV": minimum["electronic_energy_eV"]}
    result["equilibrium_identity_check"] = identity
    result["controls"] = controls
    return result


@click.command()
@click.argument("level_file", type=click.Path(exists=True, path_type=Path))
@click.argument("reactant_spectrum", type=click.Path(exists=True, path_type=Path))
@click.argument("reactant_extxyz", type=click.Path(exists=True, path_type=Path))
@click.argument("beads_json", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--index", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--write-orbit", type=click.Path(path_type=Path), help="Export a Rust level and its measured gradients to readcon.")
@click.option("--observations", type=click.Path(exists=True, path_type=Path), help="Native ordered geometry request ledger for --write-orbit.")
@click.option("--run-counts", type=click.Path(exists=True, path_type=Path), help="Native run_meta.json with provider counts for the export.")
@click.option("--confirm-same-energy-zero", is_flag=True,
              help="Confirm the eOn orbit and reactant spectrum share the PES and electronic zero.")
def main(level_file, reactant_spectrum, reactant_extxyz, beads_json, index, confirm_same_energy_zero, write_orbit, observations, run_counts):
    """Normalize a JSONL level with BEADS_JSON or an eOn lineb-orbit.con."""
    from ase.io import read

    minimum = json.loads(reactant_spectrum.read_text())
    atoms = read(reactant_extxyz, format="extxyz")
    try:
        if bool(write_orbit) != bool(observations) or (run_counts and not write_orbit):
            raise click.UsageError("supply --write-orbit and --observations together for a Rust level")
        if level_file.suffix.lower() == ".con":
            if beads_json is not None or index or write_orbit:
                raise click.UsageError("an eOn orbit contains the complete ring of one rate level")
            result = eon_orbit_rate(level_file, minimum, atoms, confirm_same_energy_zero)
        else:
            if beads_json is None:
                raise click.UsageError("a JSONL level requires BEADS_JSON")
            levels = level_file.read_text().splitlines()
            level = json.loads(levels[index])
            ring = np.array(json.loads(beads_json.read_text()), dtype=float)
            result = molecular_rate(level, minimum, atoms, ring)
            if write_orbit:
                end = sum(json.loads(row)["evaluation_counts"]["surface_calls"]["total"] for row in levels[:index+1])
                inputs = dict(levels=level_file, reactant_spectrum=reactant_spectrum,
                              reactant_geometry=reactant_extxyz, ring=beads_json, observations=observations)
                export_observed_ring(write_orbit, level, atoms, ring, observations, end, inputs, run_counts)
    except (KeyError, ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
