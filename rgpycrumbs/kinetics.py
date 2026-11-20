"""Harmonic rate analysis from stationary-point vibrational spectra."""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike


class HarmonicLogRates(NamedTuple):
    """Natural logarithms of rates in inverse seconds, with the temperature shape."""

    classical: ArrayLike
    quasi_quantum: ArrayLike


def harmonic_log_rates(
    reactant_cm: ArrayLike,
    saddle_cm: ArrayLike,
    barrier_ev: float,
    temperature_k: ArrayLike,
) -> HarmonicLogRates:
    """Compute vibrational HTST and qq-HTST rates for one reaction path.

    Spectra contain signed internal wavenumbers (cm⁻¹). Supply the electronic
    barrier in eV and a scalar or array of temperatures in kelvin. Project
    out rigid modes first. Symmetry and rotational factors are external.
    """
    from scipy.constants import Boltzmann, c, electron_volt, h

    temperature = np.asarray(temperature_k, dtype=float)
    reactant = np.asarray(reactant_cm, dtype=float)
    saddle = np.asarray(saddle_cm, dtype=float)
    if not np.all(np.isfinite(temperature) & (temperature > 0)) or not np.isfinite(barrier_ev):
        raise ValueError("expected positive finite temperature and finite electronic barrier")
    if reactant.ndim != 1 or reactant.size == 0 or reactant.shape != saddle.shape:
        raise ValueError("expected matching nonempty one-dimensional internal spectra")
    if not np.all(np.isfinite(reactant) & (reactant > 0)):
        raise ValueError("reactant wavenumbers must be positive and finite")
    if not np.all(np.isfinite(saddle) & (saddle != 0)) or np.count_nonzero(saddle < 0) != 1:
        raise ValueError("expected a finite saddle spectrum with exactly one unstable mode")

    kt = Boltzmann * temperature
    beta_barrier = barrier_ev * electron_volt / kt
    reactant_hz = 100 * c * reactant
    saddle_hz = 100 * c * saddle[saddle > 0]

    def log_partition_factors(frequencies):
        x = h * frequencies / kt[..., None]
        return (0.5 * x + np.log(-np.expm1(-x))).sum(axis=-1)

    classical = np.log(reactant_hz).sum() - np.log(saddle_hz).sum() - beta_barrier
    quantum = np.log(kt / h) + log_partition_factors(reactant_hz) - log_partition_factors(saddle_hz) - beta_barrier
    return HarmonicLogRates(classical, quantum)
