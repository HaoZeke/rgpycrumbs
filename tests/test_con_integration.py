"""Identity and periodic geometry survive the CON analysis boundary."""

import numpy as np
import pytest

readcon = pytest.importorskip("readcon", minversion="0.17.1")
pytest.importorskip("ase")

from ase import Atoms, units

from rgpycrumbs.eon.pimd_centroid import build_frames
from rgpycrumbs.geom.api.alignment import (
    AlignmentMethod,
    IRAConfig,
    _rmsd_single,
    align_structure_robust,
)


def test_centroid_keeps_reference_identity_and_cell_units(tmp_path):
    frame = readcon.ConFrame(
        cell=[10, 11, 12],
        angles=[90, 90, 90],
        atoms=[
            readcon.Atom(
                symbol="H",
                x=0,
                y=0,
                z=0,
                mass=2.014,
                atom_id=41,
                fixed=[True, False, True],
            ),
            readcon.Atom(symbol="O", x=1, y=0, z=0, mass=15.999, atom_id=7),
        ],
        metadata={
            "units": {"length": "bohr", "energy": "hartree"},
            "pbc": [True, False, True],
            "energy": -10,
        },
    )
    reference = tmp_path / "reference.con"
    readcon.write_con(str(reference), [frame])
    positions = np.array([[[0, 0, 0], [1.5, 0, 0]]])
    output = build_frames(positions, np.zeros_like(positions), [{}], reference=reference)[
        0
    ]
    assert [atom.atom_id for atom in output.atoms] == [41, 7]
    assert list(output.atoms[0].fixed) == [True, False, True]
    assert output.atoms[0].mass == pytest.approx(2.014)
    assert output.metadata["pbc"] == [True, False, True]
    np.testing.assert_allclose(output.cell, np.array([10, 11, 12]) * units.Bohr)
    np.testing.assert_allclose(output.xyz, positions[0])
    assert output.energy is None


def test_periodic_alignment_and_rmsd_use_the_cell():
    reference = Atoms(
        "HOC",
        positions=[[0.1, 1, 2], [9.9, 2, 3], [3, 4, 5]],
        cell=[10, 10, 10],
        pbc=True,
    )
    mobile = reference.copy()
    mobile.positions += [0.4, 0.2, -0.3]
    mobile.wrap()
    config = IRAConfig(enabled=False)
    assert _rmsd_single(reference, mobile, config, reference.positions) == pytest.approx(
        0, abs=1e-12
    )
    result = align_structure_robust(reference, mobile, config)
    assert result.method is AlignmentMethod.PERIODIC_TRANSLATION
    np.testing.assert_array_equal(mobile.cell, reference.cell)


def test_periodic_rmsd_keeps_open_axis_displacements():
    reference = Atoms(
        "HH", positions=[[0, 0, 0], [1, 0, 1]], cell=[10, 10, 10], pbc=[True, True, False]
    )
    mobile = reference.copy()
    mobile.positions[1, 2] += 10
    assert _rmsd_single(
        reference, mobile, IRAConfig(enabled=False), reference.positions
    ) == pytest.approx(5)
