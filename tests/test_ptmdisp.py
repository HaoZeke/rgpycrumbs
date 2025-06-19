import pytest
from click.testing import CliRunner
from ase.build import bulk
from ase.neighborlist import NeighborList
from ase.io import write
import numpy as np

from rgpycrumbs.eon.ptmdisp import main, find_mismatch_indices, CrystalStructure


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
    final_expected_indices = {
        i if i < vacancy_index else i - 1 for i in neighbor_indices
    }

    return filepath, final_expected_indices


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


def test_find_indices_on_fcc_with_defect(defect_fcc_cu_file):
    """
    On a crystal with a vacancy, it should identify the atoms neighboring the vacancy.
    """
    filepath, expected_defect_indices = defect_fcc_cu_file
    indices = find_mismatch_indices(filepath, CrystalStructure.FCC)

    assert (
        set(indices) == expected_defect_indices
    ), "Should identify all neighbors of the vacancy"


def test_find_indices_on_perfect_bcc(perfect_bcc_fe_file):
    """
    On a perfect bulk BCC crystal, searching for non-BCC atoms should return an empty list.
    """
    indices = find_mismatch_indices(perfect_bcc_fe_file, CrystalStructure.BCC)
    assert (
        len(indices) == 0
    ), "Should find no non-BCC atoms in a perfect bulk BCC crystal"


### Integration Tests for the Command-Line Interface (CLI) ###


def test_cli_quiet_output_for_defect(defect_fcc_cu_file):
    """
    Tests the default quiet CLI output. It should only print the final index list.
    """
    filepath, expected_defect_indices = defect_fcc_cu_file
    runner = CliRunner()
    result = runner.invoke(main, [str(filepath)])

    assert result.exit_code == 0, "Script should exit successfully"

    # Convert the string output to a set of integers for comparison.
    output_indices = set(map(int, result.stdout.strip().split(",")))

    assert (
        output_indices == expected_defect_indices
    ), "stdout should contain all neighbor indices"


def test_cli_structure_type_option(perfect_bcc_fe_file):
    """
    Tests that the --structure-type option correctly identifies a different crystal type.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["--structure-type", "BCC", str(perfect_bcc_fe_file)])

    assert result.exit_code == 0
    assert result.stdout.strip() == ""
