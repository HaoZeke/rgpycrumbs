"""Property calculations keep geometry, spin and stored artifacts consistent."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ase")

from ase import Atoms

from rgpycrumbs.geom._xtb import single_point


def test_single_point_and_verified_reuse(tmp_path, monkeypatch):
    binary = tmp_path / "xtb"
    binary.write_text("test executable")
    monkeypatch.setattr("rgpycrumbs.geom._xtb.shutil.which", lambda _: str(binary))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "--version" in argv:
            return SimpleNamespace(stdout="xTB test", stderr="", returncode=0)
        out = Path(kwargs["cwd"])
        (out / "wbo").write_text("1 2 .8\n")
        (out / "density.cub").write_text("retained cube\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rgpycrumbs.geom._xtb.subprocess.run", run)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    root = tmp_path / "properties"
    result = single_point(atoms, root, charge=0, multiplicity=1)
    assert "--sp" in calls[1] and "--opt" not in calls[1]
    assert calls[1][calls[1].index("--uhf") + 1] == "0"
    assert single_point(atoms, root, charge=0, multiplicity=1) == result
    assert len(calls) == 2
    (root / "wbo").write_text("1 2 1.5\n")
    with pytest.raises(ValueError, match="artifact changed"):
        single_point(atoms, root, charge=0, multiplicity=1)


def test_incompatible_electronic_state_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="electron count"):
        single_point(Atoms("H2"), tmp_path / "invalid", charge=0, multiplicity=2)
    assert not (tmp_path / "invalid").exists()
