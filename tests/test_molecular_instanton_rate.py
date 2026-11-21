import numpy as np
from ase import Atoms
from scipy import constants

from rgpycrumbs.eon.molecular_instanton_rate import molecular_rate


def test_absolute_rate_uses_explicit_ring_partition_and_cancels_energy_zero():
    atoms = Atoms("H2O", positions=[[0.75, 0.5, 0], [-0.75, 0.5, 0], [0, 0, 0]])
    p, t, hbar, kb = 12, 275.0, 0.06466, constants.value("Boltzmann constant in eV/K")
    modes = [900.0, 1800.0, 3000.0]
    minimum = {"signed_wavenumbers_cm-1": modes, "electronic_energy_eV": 0.0}
    ring = np.tile(atoms.positions, (p, 1, 1))
    level = {"controls": {"boltzmann": kb, "hbar": hbar}, "reference_energy": 0.0,
             "rate": {"temperature": t, "beads": p, "modes": 9 * p - 7, "log_k_times_Zr": -52.0}}
    time_unit = 1e-10 * np.sqrt(constants.atomic_mass / constants.electron_volt)
    quantum = 2 * np.pi * constants.c * 100 * np.array(modes) * time_unit * hbar / (p * kb * t)
    spring_terms = 4 * np.sin(np.pi * np.arange(p) / p) ** 2
    log_q = -0.5 * np.log(spring_terms[:, None] + quantum**2).sum() - 3 * np.log(spring_terms[1:]).sum()
    expected = (-52.0 - log_q - np.log(time_unit)) / np.log(10)
    actual = molecular_rate(level, minimum, atoms, ring)["log10_rate_per_s"]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    shift = -124.5
    level["reference_energy"] = minimum["electronic_energy_eV"] = shift
    level["rate"]["log_k_times_Zr"] -= shift / (kb * t)
    ring = 2 * ring + np.array([2, 3, 4])
    actual = molecular_rate(level, minimum, atoms, ring)["log10_rate_per_s"]
    np.testing.assert_allclose(actual, expected + np.log10(8), atol=1e-11)


def test_eon_full_orbit_matches_the_equivalent_flux_record(tmp_path):
    import json

    import readcon
    from ase.io import write
    from click.testing import CliRunner
    from rgpycrumbs.eon.molecular_instanton_rate import main

    atoms = Atoms("H2O", positions=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    equilibrium = tmp_path / "reactant.extxyz"
    write(equilibrium, atoms)
    p = 8
    controls = {"hbar": 0.06466, "boltzmann": 8.617333262e-5}
    lineb = {**controls, "molecular": True, "beads": p, "temperature": 275.,
             "log_rate_over_reactant": -20., "spectrum": {"negative_modes": 1, "modes": 9*p-7}}
    level = {"controls": controls, "reference_energy": -1.,
             "rate": {"beads": p, "temperature": 275., "log_k_times_Zr": -20., **lineb["spectrum"]}}
    spectrum = tmp_path / "spectrum.json"
    spectrum.write_text(json.dumps({"electronic_energy_eV": -1., "signed_wavenumbers_cm-1": [900.,1800.,3000.]}))
    ring = np.tile(atoms.positions, (p, 1, 1))
    ring[:,0,0] += 0.125*np.cos(2*np.pi*np.arange(p)/p)
    rows = tmp_path / "beads.json"
    rows.write_text(json.dumps(ring.tolist()))
    records = tmp_path / "levels.jsonl"
    records.write_text(json.dumps(level))
    con = tmp_path / "lineb-orbit.con"
    frames = []
    for i, xyz in enumerate(ring):
        frames.append(readcon.ConFrame(cell=[10.,10.,10.], angles=[90.,90.,90.],
            atoms=[readcon.Atom(symbol=a.symbol, mass=a.mass, atom_id=j+1, x=x,y=y,z=z)
                   for j,(a,(x,y,z)) in enumerate(zip(atoms,xyz))],
            metadata={"temperature_K": 275., "tau_over_beta": i/p, "lineb": lineb,
                      "lineb_orbit": {"mobile_atom_ids":[1,2,3]}}))
    readcon.write_con(str(con), frames)
    runner = CliRunner()
    args = [str(con),str(spectrum),str(equilibrium)]
    assert runner.invoke(main,args).exit_code != 0
    measured = runner.invoke(main,[*args,"--confirm-same-energy-zero"])
    assert measured.exit_code == 0, measured.output
    expected = runner.invoke(main,[str(records),str(spectrum),str(equilibrium),str(rows)])
    assert expected.exit_code == 0, expected.output
    data = json.loads(measured.output)
    assert data["energy_reference"]["independent_orbit_reference_energy_eV"] is None
    np.testing.assert_allclose(data["log10_rate_per_s"],json.loads(expected.output)["log10_rate_per_s"],atol=1e-8,rtol=0)
    readcon.write_con(str(con), frames[::-1])
    assert "phase" in runner.invoke(main,[*args,"--confirm-same-energy-zero"]).output
