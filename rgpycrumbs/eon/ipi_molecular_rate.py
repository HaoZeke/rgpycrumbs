"""Combine native i-PI partition reports into a single-path molecular rate."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "numpy", "readcon", "scipy"]
# ///

import importlib
import json
import re
from pathlib import Path

import click
import numpy as np


def fields(lines, heading):
    for index, line in enumerate(lines):
        if heading in line:
            values = re.split(r"[|()\s]+", lines[index + 1].strip())
            values = [complex(v) for v in values if v]
            if any(v.imag != 0 or not np.isfinite(v.real) for v in values):
                raise ValueError("partition factors must be finite and real")
            return [v.real for v in values]
    raise ValueError(f"missing i-PI partition table {heading}")


def molecular_rate(minimum, orbit):
    """Use the native log(Qvib*N) convention with a shared reactant energy zero."""
    constants = importlib.import_module("scipy.constants")
    rlines, ilines = minimum.splitlines(), orbit.splitlines()
    rtemp, itemp = [
        float(re.search(r"Temperature:\s+([\d.eE+-]+) K", text)[1])
        for text in (minimum, orbit)
    ]
    nbeads = int(re.search(r"Instanton rate\. Nbeads (\d+)", orbit)[1])
    nreact = int(re.search(r"Reactants\. Nbeads (\d+)", minimum)[1])
    if nreact != nbeads or not np.isclose(rtemp, itemp, rtol=1e-12) or itemp <= 0:
        raise ValueError(
            "reactant and orbit require matching temperature and full ring size"
        )
    for text in (minimum, orbit):
        if not re.search(r"ASR:\s+poly\b", text):
            raise ValueError("expected a nonlinear freely rotating molecular system")
    if "Overwriting energy shift with the provided values" not in orbit:
        raise ValueError(
            "instanton postprocessing must use the reactant electronic energy"
        )
    qt_r, qr_r, qv_r = fields(rlines, "Qtras(bohr^-3)")
    _, bn_n, qt_i, qr_i, qv_i, action, _, _ = fields(ilines, "log(Qvib*N)")
    if min(bn_n, qt_i, qr_i, qt_r, qr_r) <= 0:
        raise ValueError("the path norm and partition factors must be positive")
    kelvin = constants.Boltzmann * itemp
    temp_au = kelvin / constants.physical_constants["Hartree energy"][0]
    log = (
        np.log(kelvin / constants.h)
        + 0.5 * np.log(2 * np.pi * bn_n * temp_au)
        + np.log(qt_i / qt_r)
        + np.log(qr_i / qr_r)
        + qv_i
        - qv_r
        - action
    )
    return {
        "temperature_K": itemp,
        "beads": nbeads,
        "path_degeneracy": 1,
        "action_over_hbar": action,
        "BN_times_N_atomic_units": bn_n,
        "rotational_ratio": qr_i / qr_r,
        "translational_ratio": qt_i / qt_r,
        "log_Qvib_instanton_native": qv_i,
        "log_Qvib_reactant": qv_r,
        "ln_rate_per_s": float(log),
        "log10_rate_per_s": float(log / np.log(10)),
        "source_precision": "native i-PI partition tables (three decimal places)",
    }


# i-PI 3.3.1 UnitMap (atomic units) and instanton_postproc.save_frequencies.
IPI_UNITS = dict(kelvin=3.1668152e-6, hertz=1.5198298e-16, hz_per_cm=3e10,
                 angstrom=1.8897261, dalton=1822.8885, electronvolt=0.036749326)


def fluctuation_partition(frequencies, size, beads, beta):
    """Recover the native nonlinear-molecule fluctuation product from freq.dat."""
    modes = np.asarray(frequencies)
    if modes.shape != (size,) or not np.isfinite(modes).all() or (modes == 0).sum() != 6:
        raise ValueError("expected six rigid zeros in the full instanton frequency spectrum")
    active = modes[modes != 0]
    if (active < 0).sum() != 1 or active[0] >= 0 or np.any(np.diff(active) < 0):
        raise ValueError("expected one unstable mode followed by ordered positive modes")
    # The native rate removes d[1], the imaginary-time translation, after rigid modes.
    retained = np.abs(np.delete(active, 1)) * IPI_UNITS["hertz"] * IPI_UNITS["hz_per_cm"]
    return -np.log((beta / beads) * retained).sum() + 7 * np.log(beads), active[0], active[1]


def reconstruct(case, summary, reactant_report, instanton_report):
    """Rebuild rate factors from the measured half ring, energies and native spectra."""
    import hashlib
    import xml.etree.ElementTree as ET
    import readcon
    from ase.io import read
    from scipy import constants
    from rgpycrumbs.eon.molecular_instanton_rate import log_inertia_det

    native = case / "native"
    files = {"orbit": case / "ring.con", "observations": case / "validation-gradients.jsonl",
             "parameters": native / "parameters.json", "minimum": native / "minimum.xyz",
             "reactant_input": native / "reactant.xml",
             "reactant_eigenvalues": native / "reactant/eigenvalues_reactant.dat",
             "instanton_frequencies": native / "instanton/freq.dat"}
    params = json.loads(files["parameters"].read_text())
    p, temperature = summary["beads"], summary["temperature_K"]
    if (params["beads"] != p or params["path_degeneracy"] != 1 or
            not np.isclose(params["temperature_K"], temperature, rtol=1e-12, atol=0)):
        raise ValueError("raw inputs disagree with the native temperature, bead count or degeneracy")
    frames = sorted(readcon.iter_con(str(files["orbit"])), key=lambda f: f.metadata["neb_bead"])
    if p < 4 or 2 * len(frames) != p or [f.metadata["neb_bead"] for f in frames] != list(range(len(frames))):
        raise ValueError("expected a complete numbered i-PI half ring")
    if ("evaluation_counts" in params and
            frames[0].metadata.get("evaluation_counts") != params["evaluation_counts"]):
        raise ValueError("native partition input and half-ring evaluation ledgers differ")
    reference = params["energy_reference_eV"]
    signature = [(a.atom_id, a.symbol, a.mass) for a in frames[0].atoms]
    masses = np.asarray([a[2] for a in signature])
    if len(set(a[0] for a in signature)) != len(signature) or not np.all(np.isfinite(masses) & (masses > 0)):
        raise ValueError("expected unique atom IDs and positive finite masses")
    for frame in frames:
        if ([(a.atom_id, a.symbol, a.mass) for a in frame.atoms] != signature or
                any(any(a.fixed) for a in frame.atoms) or
                not np.isclose(frame.metadata["temperature_K"], temperature, rtol=1e-12, atol=0)):
            raise ValueError("inconsistent atom identities, constraints or temperature in half ring")
        units = frame.metadata.get("units", {})
        if units.get("length") != "angstrom" or units.get("mass") != "amu":
            raise ValueError("expected Cartesian coordinates in angstrom and masses in amu")
    minimum = read(files["minimum"])
    mass_input = ET.parse(files["reactant_input"]).find(".//initialize/masses")
    if (mass_input is None or mass_input.get("units") != "dalton" or
            minimum.get_chemical_symbols() != [a[1] for a in signature]):
        raise ValueError("reactant geometry and masses must describe the ring species in order")
    minimum_masses = np.asarray(json.loads(mass_input.text))
    if minimum_masses.shape != masses.shape or not np.allclose(minimum_masses, masses, atol=1e-7, rtol=0):
        raise ValueError("reactant and orbit isotope masses differ")
    half = np.stack([f.xyz for f in frames])
    with files["observations"].open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != len(half):
        raise ValueError("expected one energy observation per half-ring bead")
    source = params.get("evaluation_counts", {}).get("input_seed", {}).get("surface_sha256")
    observed_sources = {r.get("surface_sha256") for r in rows}
    if len(observed_sources) > 1 or (source is not None and observed_sources != {source}):
        raise ValueError("validation observations disagree on the electronic surface")
    for i, row in enumerate(rows):
        if row["atomic_numbers"] != minimum.numbers.tolist():
            raise ValueError("validation observations use different atom species or order")
        measured = np.asarray(row["positions_angstrom"]).reshape(half[i].shape)
        if not np.allclose(measured, half[i], atol=1e-8, rtol=0):
            raise ValueError(f"validation observation does not match half-ring bead {i}")
    energies = np.asarray([row["energy_eV"] for row in rows])
    if not np.isfinite(energies).all() or not np.isfinite(reference):
        raise ValueError("nonfinite energy observation or electronic reference")

    beta = 1 / (temperature * IPI_UNITS["kelvin"])
    ring = np.concatenate((half, half[::-1]))
    dx = ring - np.roll(ring, 1, axis=0)
    bn = p * np.einsum("bij,i,bij->", dx, masses, dx) * IPI_UNITS["angstrom"]**2 * IPI_UNITS["dalton"]
    if not np.isfinite(bn) or bn <= 0:
        raise ValueError("expected a finite noncollapsed orbit")
    rotation = .5 * (log_inertia_det(ring, masses) - log_inertia_det(minimum.positions, masses))
    eig = np.loadtxt(files["reactant_eigenvalues"])
    if (eig.shape != (3 * len(masses),) or not np.isfinite(eig).all() or
            (eig == 0).sum() != 6 or (eig < 0).any()):
        raise ValueError("reactant spectrum must have six rigid zeros and positive internal modes")
    springs = 4 * np.sin(np.pi * np.arange(p) / p)**2
    q_reactant = -.5 * np.log(springs + (beta / p)**2 * eig[eig > 0, None]).sum()
    q_orbit, unstable, removed = fluctuation_partition(np.loadtxt(files["instanton_frequencies"]), ring.size, p, beta)
    potential_action = beta * IPI_UNITS["electronvolt"] * np.mean(energies - reference)
    spring_action = bn / (2 * beta)
    kt = temperature * constants.Boltzmann
    # Use the conversion of the reported atomic path norm in the table normalizer.
    kelvin_au = kt / constants.physical_constants["Hartree energy"][0]
    log_rate = (np.log(kt / constants.h) + .5 * np.log(2 * np.pi * bn * kelvin_au)
                + rotation + q_orbit - q_reactant - potential_action - spring_action)
    raw = dict(beads=p, temperature_K=temperature, negative_frequency_cm1=float(unstable),
               time_translation_cm1=float(removed), positive_reactant_modes=int((eig > 0).sum()),
               BN_times_N_atomic_units=bn, rotational_ratio=np.exp(rotation), translational_ratio=1.0,
               log_Qvib_instanton_native=q_orbit, log_Qvib_reactant=q_reactant,
               potential_action_over_hbar=potential_action, spring_action_over_hbar=spring_action,
               action_over_hbar=potential_action + spring_action, ln_rate_per_s=log_rate,
               log10_rate_per_s=log_rate / np.log(10))
    raw["raw_minus_native"] = {k: raw[k] - v for k, v in summary.items()
                               if k in raw and k not in {"beads", "temperature_K"}}
    _, _, _, _, _, _, s1, s2 = fields(instanton_report.splitlines(), "log(Qvib*N)")
    raw["raw_minus_native"].update(potential_action_over_hbar=potential_action - s1,
                                   spring_action_over_hbar=spring_action - s2)
    # Each factor in the native partition tables is rounded to three decimal places.
    half_digit = 0.0005
    qt_r, qr_r, _ = fields(reactant_report.splitlines(), "Qtras(bohr^-3)")
    _, bn_tab, qt_i, qr_i, _, _, _, _ = fields(instanton_report.splitlines(), "log(Qvib*N)")
    intervals = {}
    for key in ("action_over_hbar", "BN_times_N_atomic_units", "log_Qvib_instanton_native",
                "log_Qvib_reactant", "potential_action_over_hbar", "spring_action_over_hbar"):
        rounded = {"potential_action_over_hbar": s1, "spring_action_over_hbar": s2}.get(key, summary.get(key))
        intervals[key] = (rounded - half_digit, rounded + half_digit)
    for key, numerator, denominator in (("rotational_ratio", qr_i, qr_r), ("translational_ratio", qt_i, qt_r)):
        intervals[key] = ((numerator - half_digit) / (denominator + half_digit),
                          (numerator + half_digit) / (denominator - half_digit))
    # Bound the final log rate by propagating the monotonic dependence on each printed factor.
    bounds = []
    for direction in (0, 1):
        log_bound = (np.log(kt / constants.h) + .5 * np.log(2 * np.pi * (bn_tab + (2*direction-1)*half_digit) * kelvin_au)
                     + np.log(intervals["rotational_ratio"][direction])
                     + np.log(intervals["translational_ratio"][direction])
                     + intervals["log_Qvib_instanton_native"][direction]
                     - intervals["log_Qvib_reactant"][1-direction] - intervals["action_over_hbar"][1-direction])
        bounds.append(float(log_bound / np.log(10)))
    intervals["log10_rate_per_s"] = bounds
    raw["native_rounding"] = {k: {"lower": lo, "upper": hi, "within": bool(lo <= raw[k] <= hi)}
                              for k, (lo, hi) in intervals.items()}
    raw["evaluation_counts"] = {"scope": "raw rate reconstruction", "additional_surface_requests": 0,
                                "independently_observed_energies": len(energies)}
    raw["surface_sha256"] = observed_sources.pop()
    raw["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    raw["input_sha256"].update(reactant_table=hashlib.sha256(reactant_report.encode()).hexdigest(),
                               instanton_table=hashlib.sha256(instanton_report.encode()).hexdigest())
    raw["native_units"] = {"source": "i-PI 3.3.1 UnitMap and instanton_postproc.save_frequencies", **IPI_UNITS}
    return raw


@click.command()
@click.argument("reactant", type=click.Path(exists=True, path_type=Path))
@click.argument("instanton", type=click.Path(exists=True, path_type=Path))
@click.option("--raw-case", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Reconstruct the prefactor and action from this half-ring archive.")
@click.option("--check-native-rounding", is_flag=True, help="Require every reconstructed factor to agree with the precision of the native tables.")
def main(reactant, instanton, raw_case, check_native_rounding):
    """Read native i-PI REACTANT and INSTANTON postprocessor reports."""
    if check_native_rounding and raw_case is None:
        raise click.UsageError("--check-native-rounding requires --raw-case")
    minimum, path = reactant.read_text(), instanton.read_text()
    result = molecular_rate(minimum, path)
    if raw_case is not None:
        result["raw_reconstruction"] = reconstruct(raw_case, result, minimum, path)
    click.echo(json.dumps(result, allow_nan=False))
    if check_native_rounding:
        failed = [k for k, v in result["raw_reconstruction"]["native_rounding"].items() if not v["within"]]
        if failed:
            raise click.ClickException("outside native rounding intervals: " + ", ".join(failed))


if __name__ == "__main__":
    main()
