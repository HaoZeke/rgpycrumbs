import numpy as np
import pytest

from rgpycrumbs.kinetics import log_finite_bead_partition


@pytest.mark.pure
@pytest.mark.parametrize("beads", [5, 8])
def test_partition_matches_periodic_spring_determinant(beads):
    beta_quanta = np.array([0.1, 1.0, 12.0])
    spring = 4 * np.sin(np.pi * np.arange(beads) / beads) ** 2
    eigenvalues = spring[:, None] + (beta_quanta / beads) ** 2
    assert log_finite_bead_partition(beta_quanta, beads) == pytest.approx(
        -0.5 * np.log(eigenvalues).sum(), abs=1e-12
    )


@pytest.mark.pure
def test_fixed_site_rate_is_invariant_to_electronic_energy_origin():
    from rgpycrumbs.eon.lattice_instanton_diffusivity import lattice_diffusivity

    temperature, kb, beads = 150.0, 8.617333262e-5, 8
    rate = dict(temperature=temperature, beads=beads, negative_modes=1, modes=23,
                log_k_times_Zr=-4.0)
    level = dict(rate=rate, controls=dict(hbar=0.06465, boltzmann=kb), reference_energy=0.0)
    seed = dict(active_atoms=[4], masses=[1.008], reference_energy=0.0)
    minimum = {"mobile_atoms": [4], "signed_wavenumbers_cm-1": [100, 500, 1500],
               "electronic_energy_eV": 0.0}
    initial = lattice_diffusivity(level, minimum, seed, 1.17, 4, 3)
    reference = -31000.0
    rate["log_k_times_Zr"] -= reference / (kb * temperature)
    level["reference_energy"] = seed["reference_energy"] = minimum["electronic_energy_eV"] = reference
    shifted = lattice_diffusivity(level, minimum, seed, 1.17, 4, 3)
    assert shifted["log10_diffusivity_cm2_per_s"] == pytest.approx(
        initial["log10_hop_rate_per_s"] + np.log10(4 * 1.17e-8 ** 2 / 6), abs=1e-8
    )
    minimum["mobile_atoms"] = [0]
    with pytest.raises(ValueError, match="atom indices"):
        lattice_diffusivity(level, minimum, seed, 1.17, 4, 3)
