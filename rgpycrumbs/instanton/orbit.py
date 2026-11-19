"""Diagnostics on a reflected cosine half orbit in atomic units."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrbitAudit:
    """Reconstructed coefficients and diagnostics in atomic units."""

    coefficients: np.ndarray
    action: float
    kinetic_action: float
    residual: float
    max_potential_rise: float
    reconstruction_error: float


def _reflect_transform(samples):
    return np.fft.rfft(np.concatenate((samples, samples[-2:0:-1]), axis=0), axis=0).real


def reconstruct_orbit(
    positions,
    energy,
    gradients,
    masses,
    *,
    beta,
    cosines,
    reference,
    geometry_atol=1e-8,
):
    """Reconstruct the mode-major coefficients of an ordered cosine half orbit.

    Sample row i is at imaginary time beta*i/(2*m), where m is the
    number of rows minus one. The orbit quadrature requires m >= 2*cosines.
    Positions are in Bohr, energies in Hartree, gradients in Hartree/Bohr,
    and masses in electron masses, one mass per Cartesian coordinate.
    The residual is the largest absolute mass-weighted Galerkin component.
    The energy reference sets the zero of the potential in the action.
    """
    positions, energy, gradients, masses = (
        np.asarray(value, dtype=float) for value in (positions, energy, gradients, masses)
    )
    if positions.ndim != 2 or positions.shape[1] == 0:
        msg = "positions must be a nonempty node-by-coordinate array"
        raise ValueError(msg)
    nodes, dof = positions.shape
    m = nodes - 1
    if not isinstance(cosines, (int, np.integer)) or not 2 <= cosines <= m // 2:
        msg = "the orbit grid requires at least 2 * cosines + 1 nodes"
        raise ValueError(msg)
    if (
        gradients.shape != positions.shape
        or energy.shape != (nodes,)
        or masses.shape != (dof,)
    ):
        msg = "energy, gradient and mass shapes must match the orbit"
        raise ValueError(msg)
    if (
        not all(
            np.isfinite(value).all() for value in (positions, energy, gradients, masses)
        )
        or (masses <= 0).any()
        or not np.isfinite([reference, beta, geometry_atol]).all()
        or beta <= 0
        or geometry_atol < 0
    ):
        msg = "require finite input, positive masses and beta, and nonnegative tolerance"
        raise ValueError(msg)

    coefficients = _reflect_transform(positions)[:cosines] / m
    coefficients[0] *= 0.5
    modes = np.arange(cosines)
    # Invert the reflected series on its original nodes.
    grid = np.zeros((m + 1, dof))
    grid[:cosines] = m * coefficients
    grid[0] *= 2
    recovered = np.fft.irfft(grid, n=2 * m, axis=0)[:nodes]
    error = float(np.max(np.abs(recovered - positions)))
    if error > geometry_atol:
        msg = f"samples differ from the declared cosine basis by {error:.3g} Bohr"
        raise ValueError(msg)

    kinetic_weight = ((2 * np.pi / beta * modes) ** 2 * beta / 2)[:, None] * masses
    kinetic = float(np.sum(kinetic_weight * coefficients**2) / 2)
    projected = _reflect_transform(gradients)[:cosines] * (beta / (2 * m))
    residual = (projected + kinetic_weight * coefficients) / np.sqrt(masses)
    shifted = energy - reference
    potential = (np.sum(shifted[1:-1]) + (shifted[0] + shifted[-1]) / 2) * beta / m
    return OrbitAudit(
        coefficients,
        kinetic + float(potential),
        kinetic,
        float(np.max(np.abs(residual))),
        float(np.max(shifted)),
        error,
    )


def proton_transfer_geometry(positions, cell, indices):
    """Measure a proton half-orbit in an orthorhombic cell, in angstroms."""
    positions, cell = np.asarray(positions), np.asarray(cell)
    h, donor, acceptor = indices
    if len(set(indices)) != 3 or cell.shape != (3,) or not np.all(cell > 0):
        msg = "require three different atoms and positive orthorhombic cell lengths"
        raise ValueError(msg)

    def mic(value):
        return value - cell * np.rint(value / cell)

    hydrogen = positions[:, h]
    nh = np.linalg.norm(mic(positions[:, acceptor] - hydrogen), axis=1)
    oh = np.linalg.norm(mic(positions[:, donor] - hydrogen), axis=1)
    x = nh - oh
    return {
        "receiver_minus_donor_distance_range_angstrom": [float(x.min()), float(x.max())],
        "proton_arc_angstrom": float(
            np.linalg.norm(mic(np.diff(hydrogen, axis=0)), axis=1).sum()
        ),
    }
