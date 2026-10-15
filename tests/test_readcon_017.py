# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
#
# SPDX-License-Identifier: MIT
"""readcon 0.17.1 contract: RoundTrip bytes, sections, and no corpus client.

RoundTrip bytes from ConFrameWriter on readcon-core 694e808 (version 0.17.1).
TODO: replace these with the published restart_exact golden when v0.17.1 is tagged.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

readcon = pytest.importorskip("readcon", minversion="0.17.1")
pytest.importorskip("ase")
pytest.importorskip("chemparseplot")

from chemparseplot.parse.eon.con_io import (
    read_con_frames,
    read_first_frame,
    write_con_frames,
)

from rgpycrumbs.eon.con_splitter import _write_output_frame, con_splitter
from rgpycrumbs.eon.helpers import load_profile_rows

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RESTART = FIXTURES / "restart_exact.con"
RESTART_VEL = FIXTURES / "restart_exact_vel.con"
_SUBNORMAL = struct.unpack("<d", (1).to_bytes(8, "little"))[0]

_PEP723 = (
    "rgpycrumbs/eon/plt_con.py",
    "rgpycrumbs/eon/con_splitter.py",
    "rgpycrumbs/eon/gen_dimer.py",
    "rgpycrumbs/eon/plt_kmc.py",
    "rgpycrumbs/eon/plt_min.py",
    "rgpycrumbs/eon/plt_neb.py",
    "rgpycrumbs/eon/plt_neb_stitch.py",
    "rgpycrumbs/eon/plt_saddle.py",
    "rgpycrumbs/uma/prepare_aoti.py",
    "rgpycrumbs/eon/tls_pairs.py",
    "rgpycrumbs/eon/modes_tensormap.py",
    "rgpycrumbs/eon/pimd_centroid.py",
)

_BANNED = (
    "lmdb",
    "readcon.db",
    "readcon_db",
    "ConCorpus",
    "append_trajectory",
    "canonical=True",
)


def _bits(value: float) -> bytes:
    return struct.pack("<d", float(value))


def test_restart_goldens_roundtrip_bytes(tmp_path: Path):
    for src in (RESTART, RESTART_VEL):
        frame = read_first_frame(src)
        out = tmp_path / src.name
        write_con_frames(out, [frame])
        assert out.read_bytes() == src.read_bytes()


def test_restart_exact_keeps_signed_zero_subnormal_and_time_unit():
    frame = read_first_frame(RESTART)
    coords = np.asarray(frame.xyz, dtype=np.float64).ravel()
    assert any(_bits(value) == struct.pack("<d", -0.0) for value in coords)
    assert any(_bits(value) == struct.pack("<d", _SUBNORMAL) for value in coords)
    atoms = frame.to_ase()
    assert (
        np.asarray(frame.xyz, dtype=np.float64).tobytes()
        == np.asarray(atoms.get_positions(), dtype=np.float64).tobytes()
    )
    text = RESTART_VEL.read_text()
    assert "(amu*angstrom^2/eV)^0.5" in text


def test_unmoved_split_matches_restart_bytes(tmp_path: Path):
    out = tmp_path / "split"
    result = CliRunner().invoke(
        con_splitter,
        [
            str(RESTART),
            "--images-per-path",
            "1",
            "--output-dir",
            str(out),
            "--align-type",
            "none",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "ipath_000.con").read_bytes() == RESTART.read_bytes()


def test_translation_keeps_displacement_and_force_bits(tmp_path: Path):
    frame = read_first_frame(RESTART)
    for index, atom in enumerate(frame.atoms):
        atom.dx = float(index) + 0.125
        atom.dy = -0.0
        atom.dz = _SUBNORMAL
    dx_bits = [(_bits(atom.dx), _bits(atom.dy), _bits(atom.dz)) for atom in frame.atoms]
    force_bits = [
        (_bits(atom.fx), _bits(atom.fy), _bits(atom.fz)) for atom in frame.atoms
    ]
    atoms = frame.to_ase()
    atoms.translate([0.2, -0.1, 0.05])
    dest = tmp_path / "shifted.con"
    _write_output_frame(dest, atoms, frame, None, preserve_source=False)
    back = read_first_frame(dest)
    assert back.has_displacements
    assert back.has_forces
    got_dx = [(_bits(atom.dx), _bits(atom.dy), _bits(atom.dz)) for atom in back.atoms]
    got_f = [(_bits(atom.fx), _bits(atom.fy), _bits(atom.fz)) for atom in back.atoms]
    assert got_dx == dx_bits
    assert got_f == force_bits
    assert np.array_equal(
        np.asarray(back.xyz, dtype=np.float64),
        np.asarray(atoms.get_positions(), dtype=np.float64),
    )


def test_rotation_drops_forces_and_displacements(tmp_path: Path):
    frame = read_first_frame(RESTART)
    atoms = frame.to_ase()
    positions = atoms.get_positions().copy()
    positions[0, 0] = positions[0, 0] + 0.4
    positions[1, 1] = positions[1, 1] - 0.2
    atoms.set_positions(positions)
    dest = tmp_path / "rotated.con"
    _write_output_frame(dest, atoms, frame, None, preserve_source=False)
    back = read_first_frame(dest)
    assert back.has_forces is False
    assert back.has_displacements is False


def test_load_profile_rows_reads_metadata_and_dat_columns(tmp_path: Path):
    con = tmp_path / "neb.con"
    frame = read_first_frame(RESTART)
    frame.set_scalar_metadata("neb_bead", 0)
    frame.set_scalar_metadata("reaction_coordinate", 0.25)
    frame.set_scalar_metadata("relative_energy", 1.5)
    frame.set_scalar_metadata("parallel_force", -0.2)
    write_con_frames(con, [frame])
    rows = load_profile_rows(con)
    assert rows.shape[0] == 5
    assert rows[1, 0] == pytest.approx(0.25)
    assert rows[2, 0] == pytest.approx(1.5)
    dat = tmp_path / "neb_000.dat"
    dat.write_text("img rc energy force\n0 0.0 0.0 0.0\n1 0.1 0.2 -0.3\n")
    loaded = load_profile_rows(dat)
    np.testing.assert_allclose(loaded[2], [0.0, 0.2])


def test_plt_neb_stitch_calls_stitch(monkeypatch, tmp_path: Path):
    import rgpycrumbs.eon.plt_neb_stitch as stitch_mod

    captured: dict[str, object] = {}

    summary = type(
        "_Summary",
        (),
        {
            "n_frames": 2,
            "out_dir": tmp_path,
            "boundary_indices": (0, 1),
            "segments": (),
            "highest_energy": 0.0,
            "highest_index": 0,
        },
    )()

    def _fake(seg_specs, out_dir, saddle_overrides=None):
        captured["segs"] = seg_specs
        captured["out"] = out_dir
        captured["overrides"] = saddle_overrides
        return summary

    monkeypatch.setattr(stitch_mod, "stitch_neb_segments", _fake)
    result = CliRunner().invoke(
        stitch_mod.main,
        [
            "--segment",
            "reactant to product:neb.con:0:4",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["segs"] == [("reactant to product", "neb.con", 0, 4)]
    assert captured["overrides"] == {}


def test_direction_dat_keeps_fixed_width_format():
    text = (ROOT / "rgpycrumbs" / "eon" / "seed_dimers.py").read_text()
    assert 'fmt="%18.12f"' in text


def test_pep723_headers_pin_readcon_0171():
    for rel in _PEP723:
        text = (ROOT / rel).read_text()
        assert "readcon>=0.17.1" in text, rel


def test_project_dependencies_do_not_list_readcon():
    text = (ROOT / "pyproject.toml").read_text()
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    assert "readcon" not in project


def test_library_code_does_not_open_a_corpus():
    offenders: list[str] = []
    for path in sorted((ROOT / "rgpycrumbs").rglob("*.py")):
        text = path.read_text()
        hits = [token for token in _BANNED if token in text]
        if hits:
            offenders.append(f"{path.relative_to(ROOT)}: {', '.join(hits)}")
    assert offenders == []
