import numpy as np
import pytest

from rgpycrumbs.eon.ipi_molecular_rate import molecular_rate


@pytest.mark.interpolation
def test_single_path_rate_reproduces_published_cookbook_tunnelling_factor():
    from scipy.constants import Boltzmann, h

    # The reactant partition factors cancel from the ratio k_RPI/k_TST.
    minimum = """Temperature: 300 K
ASR: poly
We are done. Reactants. Nbeads 160
Qtras(bohr^-3) | Qrot | logQvib_rp
9.298 | 438.764 | -47.337"""
    orbit = """Temperature: 300 K
ASR: poly
Overwriting energy shift with the provided values
We are done. Instanton rate. Nbeads 160 (diff only 80.0)
BN (BN*N) | Qt(bohr^-3) | Qrot | log(Qvib*N) | S/hbar ( S1/hbar,S2/hbar ) |
14.280 (2284.847) | 10.188 | 1251.028 | -43.477 | 25.026 (23.940 1.085) |"""
    log_rate_tst = (
        np.log(Boltzmann * 300 / h)
        + np.log(10.187982156996487 / 9.298)
        + np.log(1206.1516000829881 / 438.764)
        - 44.27834777466829
        + 47.337
        - 25.165639493793847
    )
    log_rate = molecular_rate(minimum, orbit)["ln_rate_per_s"]
    np.testing.assert_allclose(np.exp(log_rate - log_rate_tst), 9.81, rtol=0.002)


@pytest.mark.interpolation
def test_raw_spectrum_retains_the_unstable_mode_and_removes_time_translation():
    from rgpycrumbs.eon.ipi_molecular_rate import fluctuation_partition

    # The unstable mode can be smaller in magnitude than the time-shift mode.
    exported = np.array([0.0] * 6 + [-0.25, 0.80, 3.0, 5.0, 7.0, 9.0])
    logq, unstable, removed = fluctuation_partition(exported, 12, 4, 1000.0)
    assert (unstable, removed) == (-0.25, 0.80)
    retained = np.array([0.25, 3.0, 5.0, 7.0, 9.0])
    np.testing.assert_allclose(
        logq, -np.log(250 * retained * 3e10 * 1.5198298e-16).sum() + 7 * np.log(4)
    )
    with pytest.raises(ValueError, match="unstable"):
        fluctuation_partition(-abs(exported), 12, 4, 1000.0)
