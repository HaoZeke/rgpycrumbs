"""Reconstruct a molecular LI rate from native ORCA EnGrad observations."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "click", "numpy", "readcon", "scipy"]
# ///

import hashlib
import json
import tarfile
from pathlib import Path

import click
import numpy as np
from scipy import constants
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.sparse import bmat, block_diag, coo_matrix, csc_matrix, eye, kron
from scipy.sparse.linalg import splu


def projected_logdet(hessian, null_vectors):
    """Log absolute determinant on the orthogonal complement of the given columns."""
    q, r = np.linalg.qr(null_vectors, mode="reduced")
    if np.linalg.matrix_rank(r) != r.shape[1]:
        raise ValueError("expected independent phase and rigid directions")
    columns = csc_matrix(q)
    bordered = bmat([[hessian, columns], [columns.T, None]], format="csc")
    return float(np.log(np.abs(splu(bordered).U.diagonal())).sum())


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def image_observations(archive, launches, band, orbit, step):
    d = band[0].xyz.size
    n = len(orbit)
    energies = np.full(n, np.nan)
    energies[[0, n // 2]] = [band[0].metadata["energy"], band[-1].metadata["energy"]]
    gradients = np.full((len(band), 2, d, d), np.nan)
    eh = constants.physical_constants["Hartree energy in eV"][0]
    bohr = constants.physical_constants["Bohr radius"][0] * 1e10
    with tarfile.open(archive) as raw:
        for row in launches["physical_launches_in_order"]:
            phase = row["phase"]
            if phase not in ("orbit", "image_hessian"):
                continue
            stream = raw.extractfile(row["launch"] + "/output.engrad")
            if stream is None:
                raise ValueError("missing native EnGrad record")
            values = [float(line.split()[0]) for line in stream
                      if line.strip() and not line.lstrip().startswith(b"#")]
            if int(values[0]) != d // 3 or len(values) < 2 + d:
                raise ValueError("EnGrad atom or gradient count differs from band")
            if phase == "orbit":
                bead = row["bead"]
                if not 0 < bead < n // 2 or np.isfinite(energies[bead]):
                    raise ValueError("expected a single observed energy for each moving outbound bead")
                energies[bead] = energies[-bead] = values[1] * eh
            else:
                image, coordinate, sign = row["image"], row["coordinate"], row["sign"]
                if not (0 <= image < len(band) and 0 <= coordinate < d and sign in (-1, 1)):
                    raise ValueError("unrecognized native Hessian displacement")
                col = gradients[image, int(sign > 0), :, coordinate]
                if np.isfinite(col).any():
                    raise ValueError("duplicate native Hessian displacement")
                col[:] = np.asarray(values[2:2 + d]) * eh / bohr
    if not np.isfinite(gradients).all() or not np.isfinite(energies).all():
        raise ValueError("incomplete displaced-gradient or reflected-orbit energy observations")
    h = (gradients[:, 1] - gradients[:, 0]) / (2 * step)
    return energies, (h + h.transpose(0, 2, 1)) / 2


def path_parameters(knots, images, ring):
    curve = CubicSpline(knots, images)
    # Use a local chord neighbourhood for each bead to preserve the branch.
    grid = np.linspace(knots[0], knots[-1], max(257, 16 * len(images)))
    sampled = curve(grid)
    half = len(ring) // 2
    outgoing = [knots[0]]
    for target in ring[1:half]:
        nearest = np.argmin(np.sum((sampled - target)**2, axis=1))
        lower = grid[max(0, nearest - 1)]
        upper = grid[min(len(grid) - 1, nearest + 1)]
        fit = minimize_scalar(lambda s: np.sum((curve(s) - target)**2),
                              bounds=(lower, upper), method="bounded", options={"xatol": 1e-15})
        outgoing.append(fit.x)
    outgoing.append(knots[-1])
    parameters = np.r_[outgoing, outgoing[-2:0:-1]]
    if np.any(np.diff(outgoing) <= 0):
        raise ValueError("orbit does not follow the outward image order")
    error = np.abs(curve(parameters) - ring).max()
    return parameters, float(error)


def inertia_logdet(positions, masses):
    centered = positions - np.average(positions.mean(axis=0), axis=0, weights=masses)
    covariance = np.einsum("bij,bik,i->jk", centered, centered, masses) / len(positions)
    sign, logdet = np.linalg.slogdet(np.eye(3) * np.trace(covariance) - covariance)
    if sign <= 0:
        raise ValueError("expected a nonlinear molecule with positive principal moments")
    return logdet


@click.command()
@click.argument("case", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reactant_spectrum", type=click.Path(exists=True, path_type=Path))
@click.argument("reactant_geometry", type=click.Path(exists=True, path_type=Path))
def main(case, reactant_spectrum, reactant_geometry):
    """Read a native reflected ORCA CASE, reactant spectrum and minimum XYZ."""
    import readcon
    from ase.io import read

    inputs = {name: case / name for name in ("native.json", "absolute_rate.json", "lineb.con",
              "lineb-orbit.con", "engrad-launches.tgz", "orbit-launches.json")}
    inputs.update(reactant_spectrum=reactant_spectrum, reactant_geometry=reactant_geometry)
    hashes = {k: sha256(v) for k, v in inputs.items()}
    audit = json.loads(inputs["orbit-launches.json"].read_bytes())
    for name in ("native.json", "lineb.con", "lineb-orbit.con", "engrad-launches.tgz"):
        if audit["input_sha256"][name] != hashes[name]:
            raise ValueError(f"native launch audit does not match {name}")

    native = json.loads(inputs["native.json"].read_bytes())
    rate = native["lineb"]
    band = sorted(readcon.iter_con(str(inputs["lineb.con"])), key=lambda f: f.metadata["neb_bead"])
    orbit = sorted(readcon.iter_con(str(inputs["lineb-orbit.con"])), key=lambda f: f.metadata["tau_over_beta"])
    signature = [(a.atom_id, a.symbol, a.mass, tuple(a.fixed)) for a in band[0].atoms]
    if any([(a.atom_id, a.symbol, a.mass, tuple(a.fixed)) for a in f.atoms] != signature for f in band + orbit):
        raise ValueError("inconsistent readcon atom identities or constraints")
    masses = np.array([a.mass for a in band[0].atoms])
    n, d = len(orbit), 3 * len(masses)
    if n < 4 or n % 2 or n != rate["beads"] or not rate["molecular"] or any(any(s[3]) for s in signature):
        raise ValueError("expected an even full-period, unconstrained molecular ring")
    if rate["hessian"] != "interpolated_images" or rate["image_hessian_step_angstrom"] <= 0:
        raise ValueError("expected Hessians from displaced image gradients")
    phases = [f.metadata["tau_over_beta"] for f in orbit]
    np.testing.assert_allclose(phases, np.arange(n) / n, rtol=0, atol=1e-12)
    xyz = np.stack([f.xyz for f in orbit])
    np.testing.assert_allclose(xyz[1:n//2], xyz[:n//2:-1], rtol=0, atol=1e-12)
    minimum = read(reactant_geometry)
    spectrum = json.loads(reactant_spectrum.read_bytes())
    if minimum.get_chemical_symbols() != [a.symbol for a in band[0].atoms]:
        raise ValueError("reactant geometry has a different atom order")
    np.testing.assert_allclose(minimum.get_masses(), masses, rtol=0, atol=1e-12)
    ref = spectrum["electronic_energy_eV"]
    if abs(ref - native["reference_energy_eV"]) > 1e-9:
        raise ValueError("reactant electronic energy differs between native and harmonic inputs")

    root = np.sqrt(masses)[:, None]
    weighted = xyz * root
    images = (np.stack([f.xyz for f in band]) * root).reshape(len(band), d)
    knots = np.r_[0, np.cumsum(np.linalg.norm(np.diff(images, axis=0), axis=1))]
    parameters, fit_error = path_parameters(knots, images, weighted.reshape(n, d))
    energies, h = image_observations(inputs["engrad-launches.tgz"], audit, band, orbit,
                                     rate["image_hessian_step_angstrom"])
    mass_scale = np.repeat(root, 3)
    blocks = CubicSpline(knots, h / np.outer(mass_scale, mass_scale))(parameters)

    period, hbar = 2 * rate["half_period"], rate["hbar"]
    beta_n, spring = period / (n * hbar), (n / period)**2
    index = np.arange(n)
    shift = coo_matrix((np.ones(n), (index, (index + 1) % n)), shape=(n, n)).tocsc()
    hessian = block_diag(blocks) + spring * kron(2 * eye(n) - shift - shift.T, eye(d))
    center = xyz - np.average(xyz.mean(axis=0), axis=0, weights=masses)
    rigid = [np.broadcast_to(root * v, weighted.shape).ravel() for v in np.eye(3)]
    rigid += [(root * np.cross(v, center)).ravel() for v in np.eye(3)]
    previous = weighted - np.roll(weighted, 1, axis=0)
    central = (np.roll(weighted, -1, axis=0) - np.roll(weighted, 1, axis=0)) / 2
    adjacent_det = projected_logdet(hessian, np.column_stack([previous.ravel(), *rigid]))
    central_det = projected_logdet(hessian, np.column_stack([central.ravel(), *rigid]))
    action = beta_n * (np.sum(energies - ref) + 0.5 * spring * np.sum(previous**2))
    flux = (-0.5 * np.log(2*np.pi*beta_n*hbar**2) + np.log(np.linalg.norm(previous))
            - (n*d - 6) * np.log(beta_n*hbar) - adjacent_det/2 - action)
    time_unit = 1e-10 * np.sqrt(constants.atomic_mass / constants.electron_volt)
    frequencies = np.array(spectrum["signed_wavenumbers_cm-1"])
    if len(frequencies) != d-6 or not np.all(frequencies > 0):
        raise ValueError("expected 3N−6 positive reactant frequencies")
    z = 2*np.pi*constants.c*100*frequencies*hbar*time_unit / (rate["boltzmann"]*rate["temperature"])
    ln_q = -np.sum(np.log(2*np.sinh(n*np.arcsinh(z/(2*n)))))
    rotation = (inertia_logdet(xyz, masses) - inertia_logdet(minimum.positions[None], masses))/2
    result = (flux - ln_q + 6*np.log(n) + rotation - np.log(time_unit))/np.log(10)
    reference_rate = json.loads(inputs["absolute_rate.json"].read_bytes())["log10_rate_per_s"]
    click.echo(json.dumps(dict(temperature_K=rate["temperature"], beads=n, path_images=len(band),
        additional_surface_requests=0, displaced_gradients=2*len(band)*d, path_degeneracy=1,
        input_sha256=hashes, max_path_fit_error_angstrom_sqrt_u=fit_error,
        adjacent_projected_log_abs_determinant=adjacent_det, centered_projected_log_abs_determinant=central_det,
        centered_minus_adjacent_log10_rate=(np.log(np.linalg.norm(central)/np.linalg.norm(previous))
            -(central_det-adjacent_det)/2)/np.log(10), action_over_hbar=action,
        log_flux_reactant_electronic_zero=flux, log_Q_vib=ln_q, log_rotation_ratio=rotation,
        log10_rate_per_s=result, native_log10_rate_per_s=reference_rate,
        reconstructed_minus_native_decades=result-reference_rate), allow_nan=False))


if __name__ == "__main__":
    main()
