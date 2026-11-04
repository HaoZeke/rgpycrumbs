# TODO(rg): this needs some work still
from tests.conftest import skip_if_not_env

skip_if_not_env("ptm")

import gzip
import inspect
import itertools
import warnings

import numpy as np
import pytest
import readcon
from ase.build import bulk
from ase.io import write
from ase.neighborlist import NeighborList
from click.testing import CliRunner
from matplotlib.image import imread
from ovito import scene
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource

pytestmark = pytest.mark.ptm

from rgpycrumbs.eon.ptmdisp import (
    CrystalStructure,
    find_mismatch_indices,
    main,
)

# The first eight bytes of every PNG file.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

SELECTION_RADIUS = (
    inspect.signature(find_mismatch_indices).parameters["selection_radius"].default
)

# Sites of the 3x3x3 copper supercell in units of the lattice constant: one
# inside the cell, and one each on a face, an edge and a corner of it.
VACANCY_SITES = {
    "inside": (1, 1.5, 1.5),
    "face": (1, 0, 1),
    "edge": (0, 0, 1),
    "corner": (0, 0, 0),
}


def nearest_image_distances(atoms, point):
    """The distance from *point* to the nearest image of each atom.

    The images lie in the cell and in its neighbors along the periodic axes,
    so the distance is exact up to one cell edge.
    """
    steps = [(-1, 0, 1) if periodic else (0,) for periodic in atoms.pbc]
    shifts = np.array(list(itertools.product(*steps))) @ atoms.cell.array
    images = atoms.positions[:, None] + shifts
    return np.linalg.norm(images - point, axis=-1).min(axis=1)


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


@pytest.fixture
def selection_image(tmp_path):
    """The file that a test renders the selected atoms to."""
    return tmp_path / "selected-atoms.png"


@pytest.fixture
def empty_scene():
    """The OVITO scene of the process, without pipelines during the test."""
    kept = list(scene.pipelines)
    scene.pipelines.clear()
    yield scene
    scene.pipelines.clear()
    scene.pipelines.extend(kept)


def write_vacancy(filepath, element, lattice_constant):
    """Writes the 3x3x3 FCC supercell of *element* without its atom 40 and returns the path."""
    atoms = bulk(element, "fcc", a=lattice_constant, cubic=True) * (3, 3, 3)
    del atoms[40]
    write(filepath, atoms)
    return filepath


def copper_slab():
    """The 3x3x3 copper supercell without atom 40, as a slab normal to z.

    6 angstroms of vacuum lie above and below it, and the cell is periodic
    along x and y alone.
    """
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)
    del atoms[40]
    atoms.center(vacuum=6.0, axis=2)
    atoms.pbc = (True, True, False)
    return atoms


@pytest.fixture
def sphere_pbc(monkeypatch):
    """The periodic flags of the atoms that each call of _sphere_indices gets."""
    from rgpycrumbs.eon.ptmdisp import _sphere_indices

    seen = []

    def record(atoms, defect, radius):
        seen.append(atoms.pbc.tolist())
        return _sphere_indices(atoms, defect, radius)

    monkeypatch.setattr("rgpycrumbs.eon.ptmdisp._sphere_indices", record)
    return seen


### Unit Tests for the Core Logic ###


def test_find_indices_on_perfect_fcc(perfect_fcc_cu_file):
    """
    On a perfect bulk FCC crystal, searching for non-FCC atoms should return an empty list.
    """
    indices = find_mismatch_indices(perfect_fcc_cu_file, CrystalStructure.FCC)
    assert len(indices) == 0, "Should find no defects in a perfect bulk crystal"


@pytest.mark.parametrize("remove_fcc_vacancy", [False, True])
def test_find_indices_without_defect_atoms_returns_quietly(
    perfect_fcc_cu_file, remove_fcc_vacancy, selection_image, tmp_path, monkeypatch
):
    """With no defect atom, nothing is averaged and no selection image is drawn."""
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        indices = find_mismatch_indices(
            perfect_fcc_cu_file,
            CrystalStructure.FCC,
            remove_fcc_vacancy,
            view_selection=True,
            selection_image=selection_image,
        )
    assert indices.size == 0
    assert [path.name for path in tmp_path.iterdir()] == ["perfect_fcc.xyz"]


def test_find_indices_on_fcc_with_defect(defect_fcc_cu_file):
    """
    On a crystal with a vacancy, PTM should detect atoms near the defect.
    The exact indices depend on centrosymmetry thresholds and OVITO version,
    so we only verify the function runs and returns an array.
    """
    filepath, expected_defect_indices = defect_fcc_cu_file
    indices = find_mismatch_indices(filepath, CrystalStructure.FCC)
    assert isinstance(indices, np.ndarray)


def test_find_indices_renders_nothing_by_default(
    defect_fcc_cu_file, tmp_path, monkeypatch
):
    """Without view_selection, the defect atoms come back and no image is written."""
    filepath, expected_defect_indices = defect_fcc_cu_file
    monkeypatch.chdir(tmp_path)
    indices = find_mismatch_indices(filepath, CrystalStructure.FCC)
    assert expected_defect_indices <= set(indices.tolist())
    assert [path.name for path in tmp_path.iterdir()] == ["defect_fcc.xyz"]


def test_view_selection_renders_to_selection_image(
    defect_fcc_cu_file, selection_image, tmp_path, monkeypatch
):
    """With view_selection, the selected atoms go to selection_image alone."""
    filepath, _ = defect_fcc_cu_file
    monkeypatch.chdir(tmp_path)
    find_mismatch_indices(
        filepath,
        CrystalStructure.FCC,
        view_selection=True,
        selection_image=selection_image,
    )
    assert selection_image.read_bytes().startswith(PNG_SIGNATURE)
    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == ["defect_fcc.xyz", selection_image.name]


def test_view_selection_renders_the_same_image_again_in_one_process(
    tmp_path, empty_scene
):
    """Each render removes its pipeline, so a structure renders alike after another one."""
    nickel = write_vacancy(tmp_path / "vacancy-Ni.xyz", "Ni", 3.52)
    copper = write_vacancy(tmp_path / "vacancy-Cu.xyz", "Cu", 3.6)
    images = [
        tmp_path / name for name in ("nickel.png", "copper.png", "nickel-again.png")
    ]
    for filepath, image in zip([nickel, copper, nickel], images, strict=True):
        find_mismatch_indices(
            filepath, CrystalStructure.FCC, view_selection=True, selection_image=image
        )
    first, again = imread(images[0]), imread(images[2])
    differing = np.any(first != again, axis=-1).mean()
    assert differing == 0, (
        f"the second nickel image differs in {differing:.2%} of its pixels"
    )
    assert len(empty_scene.pipelines) == 0


def test_view_selection_keeps_the_pipelines_of_the_scene(
    defect_fcc_cu_file, selection_image, empty_scene
):
    """A render removes the pipeline that it added and no other."""
    filepath, _ = defect_fcc_cu_file
    data = ase_to_ovito(bulk("Cu", "fcc", a=3.6, cubic=True))
    other = Pipeline(source=StaticSource(data=data))
    other.add_to_scene()
    find_mismatch_indices(
        filepath,
        CrystalStructure.FCC,
        view_selection=True,
        selection_image=selection_image,
    )
    assert list(empty_scene.pipelines) == [other]


def test_find_indices_takes_the_same_sphere_wherever_the_vacancy_sits(tmp_path):
    """The sphere takes the minimum image, so a vacancy selects alike at every site."""
    a = 3.6
    supercell = bulk("Cu", "fcc", a=a, cubic=True) * (3, 3, 3)
    found, expected = {}, {}
    for where, site in VACANCY_SITES.items():
        position = a * np.array(site)
        atoms = supercell[~np.isclose(supercell.positions, position).all(axis=1)]
        filepath = tmp_path / f"vacancy-{where}.xyz"
        write(filepath, atoms)
        found[where] = set(find_mismatch_indices(filepath, CrystalStructure.FCC).tolist())
        near = nearest_image_distances(atoms, position) < SELECTION_RADIUS
        expected[where] = set(np.flatnonzero(near).tolist())
    counts = {where: len(indices) for where, indices in found.items()}
    assert counts == {where: len(indices) for where, indices in expected.items()}, (
        f"atoms returned at each site: {counts}"
    )
    assert found == expected


@pytest.mark.parametrize(
    "pbc",
    [(True, True, True), (True, True, False), (False, False, False)],
    ids=["periodic-xyz", "periodic-xy", "not-periodic"],
)
def test_sphere_takes_images_along_the_periodic_axes_alone(pbc):
    """Around the corner atom, the sphere reaches across the periodic faces alone."""
    from rgpycrumbs.eon.ptmdisp import _sphere_indices

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)
    atoms.pbc = pbc
    corner = np.flatnonzero(np.isclose(atoms.positions, 0).all(axis=1))
    sphere = _sphere_indices(atoms, corner, SELECTION_RADIUS)
    near = nearest_image_distances(atoms, atoms.positions[corner[0]]) < SELECTION_RADIUS
    assert sphere.tolist() == np.flatnonzero(near).tolist()


@pytest.mark.parametrize("name", ["slab.xyz", "slab.con"])
def test_slab_keeps_the_periodic_axes_of_its_file(name, sphere_pbc, tmp_path):
    """A slab file periodic along x and y alone reaches the sphere with those flags.

    The file goes through find_mismatch_indices and through the command.
    """
    filepath = tmp_path / name
    atoms = copper_slab()
    if filepath.suffix == ".con":
        readcon.write_con(str(filepath), [readcon.ConFrame.from_ase(atoms)])
    else:
        write(filepath, atoms, format="extxyz")
    find_mismatch_indices(filepath, CrystalStructure.FCC)
    result = CliRunner().invoke(main, ["--no-fcc-vacancy", str(filepath)])
    assert result.exit_code == 0, result.output
    expected = [[True, True, False], [True, True, False]]
    assert sphere_pbc == expected, f"the sphere got {sphere_pbc}"


@pytest.mark.parametrize("name", ["slab.con", "slab.con.gz"])
def test_con_file_without_flags_is_periodic_along_every_axis(name, sphere_pbc, tmp_path):
    """A .con file without flags reaches the sphere periodic along x, y and z."""
    plain = tmp_path / "slab.con"
    write(plain, copper_slab(), format="eon")
    filepath = tmp_path / name
    if filepath != plain:
        filepath.write_bytes(gzip.compress(plain.read_bytes()))
    find_mismatch_indices(filepath, CrystalStructure.FCC)
    assert sphere_pbc == [[True, True, True]], f"the sphere got {sphere_pbc}"


def test_file_without_a_cell_reaches_the_sphere_without_periodic_axes(
    sphere_pbc, tmp_path
):
    """A plain .xyz file has no cell, so the structure analysis runs on free atoms."""
    filepath = tmp_path / "cluster.xyz"
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)
    del atoms[40]
    write(filepath, atoms, format="xyz")
    indices = find_mismatch_indices(filepath, CrystalStructure.FCC)
    assert indices.size > 0
    assert sphere_pbc == [[False, False, False]], f"the sphere got {sphere_pbc}"


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


def test_cli_renders_nothing_by_default(frenkel_pair_cu_file, tmp_path, monkeypatch):
    """With the interstitial of a Frenkel pair to render, the command writes no image."""
    filepath, _, _ = frenkel_pair_cu_file
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, [str(filepath)])
    assert result.exit_code == 0, result.output
    assert [path.name for path in tmp_path.iterdir()] == ["frenkel_pair.xyz"]


def test_cli_selection_image_renders_the_selected_atoms(
    frenkel_pair_cu_file, selection_image, tmp_path, monkeypatch
):
    """--selection-image renders the atoms around the defect to that file alone."""
    filepath, _, _ = frenkel_pair_cu_file
    monkeypatch.chdir(tmp_path)
    options = ["--selection-image", str(selection_image)]
    result = CliRunner().invoke(main, [*options, str(filepath)])
    assert result.exit_code == 0, result.output
    assert selection_image.read_bytes().startswith(PNG_SIGNATURE)
    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == ["frenkel_pair.xyz", selection_image.name]


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
