import numpy as np
import pytest

pytestmark = pytest.mark.pure


def test_nebpath_namedtuple():
    from rgpycrumbs.basetypes import nebpath

    p = nebpath(norm_dist=0.5, arc_dist=1.2, energy=-0.74)
    assert p.norm_dist == 0.5
    assert p.arc_dist == 1.2
    assert p.energy == -0.74


def test_nebiter_namedtuple():
    from rgpycrumbs.basetypes import nebiter, nebpath

    p = nebpath(norm_dist=0.0, arc_dist=0.0, energy=0.0)
    it = nebiter(iteration=3, nebpath=p)
    assert it.iteration == 3
    assert it.nebpath is p


def test_dimer_opt_defaults():
    from rgpycrumbs.basetypes import DimerOpt

    d = DimerOpt()
    assert d.saddle == "dimer"
    assert d.rot == "lbfgs"
    assert d.trans == "lbfgs"


def test_dimer_opt_custom():
    from rgpycrumbs.basetypes import DimerOpt

    d = DimerOpt(saddle="lanczos", rot="cg", trans="fire")
    assert d.saddle == "lanczos"


def test_spin_id():
    from rgpycrumbs.basetypes import SpinID

    s = SpinID(mol_id=42, spin="triplet")
    assert s.mol_id == 42
    assert s.spin == "triplet"


def test_mol_geom():
    from rgpycrumbs.basetypes import MolGeom

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    m = MolGeom(pos=pos, energy=-1.5, forces=forces)
    assert m.energy == -1.5
    assert m.pos.shape == (2, 3)
    assert m.forces.shape == (2, 3)


def test_saddle_measure_defaults():
    from rgpycrumbs.basetypes import SaddleMeasure

    s = SaddleMeasure()
    assert s.pes_calls == 0
    assert s.success is False
    assert s.method == "not run"
    assert np.isnan(s.saddle_energy)
    assert s.termination_status == "not set"


def test_saddle_measure_custom():
    from rgpycrumbs.basetypes import SaddleMeasure

    s = SaddleMeasure(
        pes_calls=100,
        success=True,
        method="dimer",
        saddle_energy=-0.5,
        barrier=0.3,
    )
    assert s.pes_calls == 100
    assert s.success is True
    assert s.saddle_energy == -0.5
    assert s.barrier == 0.3
