"""Harmonic rate analysis from stationary-point vibrational spectra."""

import math
from collections.abc import Sequence
from typing import NamedTuple


class HarmonicLogRates(NamedTuple):
    """Natural logarithms of vibrational transition-state rates in inverse seconds."""

    classical: float
    quasi_quantum: float


def harmonic_log_rates(
    reactant_cm: Sequence[float],
    saddle_cm: Sequence[float],
    barrier_ev: float,
    temperature_k: float,
) -> HarmonicLogRates:
    """Compute classical and quasi-quantum rates for one reaction path.

    Supply the electronic barrier in eV and the signed internal wavenumbers
    in inverse centimetres, with one negative mode at the saddle. Remove
    rigid modes before calling. Symmetry and rotational factors are external.
    """
    if not math.isfinite(temperature_k) or temperature_k <= 0 or not math.isfinite(barrier_ev):
        raise ValueError("expected positive finite temperature and finite electronic barrier")
    if len(reactant_cm) != len(saddle_cm) or not len(reactant_cm):
        raise ValueError("expected matching nonempty internal spectra")
    if any(not math.isfinite(v) or v <= 0 for v in reactant_cm):
        raise ValueError("reactant wavenumbers must be positive and finite")
    if any(not math.isfinite(v) or v == 0 for v in saddle_cm) or sum(v < 0 for v in saddle_cm) != 1:
        raise ValueError("expected a finite saddle spectrum with exactly one unstable mode")

    c_cm_s = 29979245800.0
    h_ev_s = 4.135667696923859e-15
    kt = 8.617333262145179e-5 * temperature_k
    reactant = [c_cm_s * v for v in reactant_cm]
    stable = [c_cm_s * v for v in saddle_cm if v > 0]

    def log_factor(frequency: float) -> float:
        x = h_ev_s * frequency / kt
        return x / 2 + math.log(-math.expm1(-x))

    classical = math.fsum(map(math.log, reactant)) - math.fsum(map(math.log, stable))
    quantum = math.log(kt / h_ev_s) + math.fsum(map(log_factor, reactant)) - math.fsum(map(log_factor, stable))
    return HarmonicLogRates(classical - barrier_ev / kt, quantum - barrier_ev / kt)
