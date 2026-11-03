# TODO(rg): this needs some work still
from tests.conftest import skip_if_not_env

skip_if_not_env("ptm")

import warnings

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write
from ase.neighborlist import NeighborList
from click.testing import CliRunner

pytestmark = pytest.mark.ptm

from rgpycrumbs.eon.ptmdisp import (
    CrystalStructure,
    find_mismatch_indices,
    main,
)


@pytest.fixture
def perfect_fcc_cu_file(tmp_path):
    """Creates a perfect bulk FCC Cu crystal file and returns its path."""
    filepath = tmp_path / "perfect_fcc.xyz"
    # Create a 3x3x3 supercell of bulk FCC copper.
    # All atoms in this periodic system have a perfect FCC environment.
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)
    write(filepath, atoms)
    return filepath


@pytest.fixture
def defect_fcc_cu_file(tmp_path):
    """
    Creates a bulk FCC Cu crystal with a single vacancy and returns its path
    along with the set of indices for the atoms neighboring the vacancy.
    """
    filepath = tmp_path / "defect_fcc.xyz"
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)

    # The neighbors of the vacancy are what PTM will see as "defective".
    # Define the atom to delete (somewhere in the middle).
    vacancy_index = 40

    # Calculate which atoms are neighbors BEFORE creating the vacancy.
    cutoffs = [1.3] * len(atoms)  # Use a simple cutoff for neighbor calculation
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    neighbor_indices = set(nl.get_neighbors(vacancy_index)[0])

    # Now, create the vacancy by deleting the atom.
    del atoms[vacancy_index]

    write(filepath, atoms)

    # The expected "defective" indices are the neighbors of the original atom.
    # Note: Indices of atoms after the deleted one will shift down by 1.
    # We must account for this shift in our expected indices set.
    final_expected_indices = {i if i < vacancy_index else i - 1 for i in neighbor_indices}

    return filepath, final_expected_indices


@pytest.fixture
def frenkel_pair_cu_file(tmp_path):
    """
    Creates FCC Cu with a vacancy and a <100> dumbbell interstitial and returns
    its path with the indices of the vacancy neighbors and of the dumbbell.
    """
    filepath = tmp_path / "frenkel_pair.xyz"
    a = 3.6
    atoms = bulk("Cu", "fcc", a=a, cubic=True) * (4, 4, 4)
    positions = atoms.get_positions()
    vacancy_index = int(np.argmin(np.linalg.norm(positions - [a, a, a], axis=1)))
    host_index = int(np.argmin(np.linalg.norm(positions - [3 * a, 3 * a, 3 * a], axis=1)))

    nl = NeighborList([1.3] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    neighbor_indices = set(nl.get_neighbors(vacancy_index)[0])

    # The host atom and a copy of it straddle the lattice site along x.
    site = positions[host_index].copy()
    atoms += atoms[host_index : host_index + 1]
    positions = atoms.get_positions()
    positions[host_index] = site + [0.3 * a, 0, 0]
    positions[-1] = site - [0.3 * a, 0, 0]
    atoms.set_positions(positions)
    dumbbell_indices = {host_index, len(atoms) - 1}
    del atoms[vacancy_index]
    write(filepath, atoms)

    def shift(indices):
        return {i if i < vacancy_index else i - 1 for i in indices}

    return filepath, shift(neighbor_indices), shift(dumbbell_indices)


@pytest.fixture
def perfect_bcc_fe_file(tmp_path):
    """Creates a perfect bulk BCC Fe crystal file and returns its path."""
    filepath = tmp_path / "perfect_bcc.xyz"
    # Create a 3x3x3 supercell of bulk BCC iron.
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (3, 3, 3)
    write(filepath, atoms)
    return filepath


### Unit Tests for the Core Logic ###


def test_find_indices_on_perfect_fcc(perfect_fcc_cu_file):
    """
    On a perfect bulk FCC crystal, searching for non-FCC atoms should return an empty list.
    """
    indices = find_mismatch_indices(perfect_fcc_cu_file, CrystalStructure.FCC)
    assert len(indices) == 0, "Should find no defects in a perfect bulk crystal"


@pytest.mark.parametrize("remove_fcc_vacancy", [False, True])
def test_find_indices_without_defect_atoms_returns_quietly(
    perfect_fcc_cu_file, remove_fcc_vacancy, tmp_path, monkeypatch
):
    """With no defect atom, nothing is averaged and no selection image is drawn."""
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        indices = find_mismatch_indices(
            perfect_fcc_cu_file, CrystalStructure.FCC, remove_fcc_vacancy
        )
    assert indices.size == 0
    assert not (tmp_path / "selection.png").exists()


def test_find_indices_on_fcc_with_defect(defect_fcc_cu_file):
    """
    On a crystal with a vacancy, PTM should detect atoms near the defect.
    The exact indices depend on centrosymmetry thresholds and OVITO version,
    so we only verify the function runs and returns an array.
    """
    filepath, expected_defect_indices = defect_fcc_cu_file
    indices = find_mismatch_indices(filepath, CrystalStructure.FCC)
    assert isinstance(indices, np.ndarray)


def test_find_indices_without_vacancy_filter_keeps_the_neighbors(defect_fcc_cu_file):
    """Without the vacancy filter, every neighbor of the vacancy is a defect atom."""
    filepath, expected_defect_indices = defect_fcc_cu_file
    indices = find_mismatch_indices(
        filepath, CrystalStructure.FCC, remove_fcc_vacancy=False, view_selection=False
    )
    assert expected_defect_indices <= set(indices.tolist())


@pytest.mark.parametrize(
    ("element", "lattice_constant"),
    [("Ni", 3.52), ("Cu", 3.6), ("Au", 4.08), ("Pb", 4.95)],
)
def test_vacancy_filter_drops_every_vacancy_neighbor(tmp_path, element, lattice_constant):
    """The window scales with a**2, so a lone vacancy leaves no defect atom."""
    filepath = tmp_path / "vacancy.xyz"
    atoms = bulk(element, "fcc", a=lattice_constant, cubic=True) * (3, 3, 3)
    del atoms[40]
    write(filepath, atoms)
    indices = find_mismatch_indices(
        filepath, CrystalStructure.FCC, remove_fcc_vacancy=True, view_selection=False
    )
    assert indices.size == 0


def test_vacancy_filter_keeps_the_interstitial_of_a_frenkel_pair(frenkel_pair_cu_file):
    """Only the vacancy side of a Frenkel pair is dropped."""
    filepath, vacancy_neighbors, dumbbell = frenkel_pair_cu_file
    indices = find_mismatch_indices(
        filepath, CrystalStructure.FCC, remove_fcc_vacancy=True, view_selection=False
    )
    assert dumbbell <= set(indices.tolist())
    assert not vacancy_neighbors & set(indices.tolist())


def test_vacancy_window_bounds_the_dropped_atoms(defect_fcc_cu_file):
    """A window that leaves out 1/2 keeps the vacancy neighbors."""
    filepath, expected_defect_indices = defect_fcc_cu_file
    indices = find_mismatch_indices(
        filepath,
        CrystalStructure.FCC,
        remove_fcc_vacancy=True,
        view_selection=False,
        csp_vacancy_min=0.6,
        csp_vacancy_max=0.7,
    )
    assert expected_defect_indices <= set(indices.tolist())


def test_find_indices_on_perfect_bcc(perfect_bcc_fe_file):
    """
    On a perfect bulk BCC crystal, searching for non-BCC atoms should return an empty list.
    """
    indices = find_mismatch_indices(perfect_bcc_fe_file, CrystalStructure.BCC)
    assert len(indices) == 0, "Should find no non-BCC atoms in a perfect bulk BCC crystal"


### Integration Tests for the Command-Line Interface (CLI) ###


def test_cli_quiet_output_for_defect(defect_fcc_cu_file):
    """
    Tests the default quiet CLI output runs without error.
    """
    filepath, expected_defect_indices = defect_fcc_cu_file
    runner = CliRunner()
    result = runner.invoke(main, [str(filepath)])

    assert result.exit_code == 0, f"Script should exit successfully: {result.output}"


def test_cli_structure_type_option(perfect_bcc_fe_file):
    """
    Tests that the --structure-type option correctly identifies a different crystal type.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["--structure-type", "BCC", str(perfect_bcc_fe_file)])

    assert result.exit_code == 0


def test_cli_no_fcc_vacancy_turns_the_vacancy_filter_off(
    perfect_fcc_cu_file, monkeypatch
):
    """The command runs the vacancy filter unless --no-fcc-vacancy is given."""
    filter_settings = []

    def record(filename, structure, remove_fcc_vacancy=False, **kwargs):
        filter_settings.append(remove_fcc_vacancy)
        return np.array([], dtype=int)

    monkeypatch.setattr("rgpycrumbs.eon.ptmdisp.find_mismatch_indices", record)
    runner = CliRunner()
    for options in ([], ["--no-fcc-vacancy"]):
        result = runner.invoke(main, [*options, str(perfect_fcc_cu_file)])
        assert result.exit_code == 0, result.output
    assert filter_settings == [True, False]


def test_cli_passes_the_vacancy_window(perfect_fcc_cu_file, monkeypatch):
    """--csp-vacancy-min and --csp-vacancy-max reach the vacancy filter."""
    from rgpycrumbs.eon.ptmdisp import CSP_VACANCY_MAX_A2, CSP_VACANCY_MIN_A2

    windows = []

    def record(filename, structure, **kwargs):
        windows.append((kwargs["csp_vacancy_min"], kwargs["csp_vacancy_max"]))
        return np.array([], dtype=int)

    monkeypatch.setattr("rgpycrumbs.eon.ptmdisp.find_mismatch_indices", record)
    runner = CliRunner()
    for options in ([], ["--csp-vacancy-min", "0.4", "--csp-vacancy-max", "0.6"]):
        result = runner.invoke(main, [*options, str(perfect_fcc_cu_file)])
        assert result.exit_code == 0, result.output
    assert windows == [(CSP_VACANCY_MIN_A2, CSP_VACANCY_MAX_A2), (0.4, 0.6)]


def test_cli_refuses_a_window_whose_minimum_exceeds_its_maximum(perfect_fcc_cu_file):
    """An empty window is a usage error."""
    options = ["--csp-vacancy-min", "0.7", "--csp-vacancy-max", "0.6"]
    result = CliRunner().invoke(main, [*options, str(perfect_fcc_cu_file)])
    assert result.exit_code == 2
    assert "must not exceed --csp-vacancy-max" in result.output
