# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for rgpycrumbs.eon.seed_dimers (gen-dimer seeding from NEB peaks)."""

from pathlib import Path

import numpy as np
import pytest

from tests._optional_imports import has_module_spec

pytestmark = pytest.mark.skipif(
    not all(has_module_spec(m) for m in ("readcon", "ase", "scipy")),
    reason="seed_dimers needs readcon + ase + scipy",
)

_TEST_NEB = Path(__file__).resolve().parent / "data" / "neb_test" / "neb.con"


def _make_band(tmp_path, energies):
    """Write a synthetic neb.con whose images carry the given energies.

    Reuses real eOn frames from the bundled test band (which keep valid
    multi-frame headers). ``readcon.ConFrame.energy`` is read-only, so the
    profile is supplied via a sibling ``neb.dat`` (eOn layout, col index 2).
    """
    from readcon import read_con, write_con

    src = read_con(str(_TEST_NEB))
    assert len(src) >= len(energies), "bundled test band too short"
    frames = src[: len(energies)]
    neb_con = tmp_path / "neb.con"
    write_con(str(neb_con), frames)
    # img  rx  energy  ...
    dat_lines = [
        f"{i}  {float(i)}  {float(e)}  0.0  0.0\n" for i, e in enumerate(energies)
    ]
    (tmp_path / "neb.dat").write_text("".join(dat_lines))
    return neb_con


def test_two_maxima_create_two_seed_dirs(tmp_path):
    from rgpycrumbs.eon.seed_dimers import seed_dimers_from_peaks

    # 5-image band, double-hump profile: maxima at indices 1 and 3.
    energies = [0.0, 1.0, 0.2, 0.9, 0.0]
    neb_con = _make_band(tmp_path, energies)
    out_dir = tmp_path / "seeds"

    seed_dirs, summary = seed_dimers_from_peaks(
        neb_con=neb_con,
        out_dir=out_dir,
        settings_name="nwchem_blyp.nwi",
        socket="nwsock",
        prominence=0.1,
    )

    assert len(seed_dirs) == 2
    assert [s.image_index for s in summary] == [1, 3]
    for seed in seed_dirs:
        assert (seed / "pos.con").is_file()
        assert (seed / "direction.dat").is_file()
        assert (seed / "displacement.con").is_file()
        assert (seed / "config.ini").is_file()

    # direction.dat is a normalized (natoms, 3) mode.
    from readcon import read_con

    natoms = len(read_con(str(neb_con))[0].to_ase())
    mode = np.loadtxt(seed_dirs[0] / "direction.dat")
    assert mode.shape == (natoms, 3)
    assert np.isclose(np.linalg.norm(mode), 1.0, atol=1e-6)

    # config.ini carries the dimer saddle-search contract.
    cfg = (seed_dirs[0] / "config.ini").read_text()
    assert "job = saddle_search" in cfg
    assert "min_mode_method = dimer" in cfg
    assert "potential = SocketNWChem" in cfg
    assert "nwchem_settings = nwchem_blyp.nwi" in cfg
    assert "unix_socket_path = nwsock" in cfg

    # energy_vs_reactant uses image 0 as the reference.
    assert summary[0].energy_vs_reactant == pytest.approx(1.0)
    assert summary[0].mode_source == "neb_tangent"


def test_prefers_matching_eon_peak_mode(tmp_path):
    from ase import Atoms
    from readcon import ConFrame, write_con

    from rgpycrumbs.eon.seed_dimers import seed_dimers_from_peaks

    energies = [0.0, 1.0, 0.0]
    neb_con = _make_band(tmp_path, energies)

    # eOn peak files: peak00 geometry matches NEB image 1 exactly.
    peak_dir = tmp_path / "peaks"
    peak_dir.mkdir()
    from readcon import read_con

    peak_atoms = read_con(str(neb_con))[1].to_ase()
    write_con(str(peak_dir / "peak00_pos.con"), [ConFrame.from_ase(peak_atoms)])
    custom_mode = np.tile([1.0, 0.0, 0.0], (len(peak_atoms), 1))
    np.savetxt(peak_dir / "peak00_mode.dat", custom_mode, fmt="%.6f")

    seed_dirs, summary = seed_dimers_from_peaks(
        neb_con=neb_con,
        out_dir=tmp_path / "seeds",
        settings_name="s.nwi",
        socket="sock",
        peak_files_dir=peak_dir,
        prominence=0.1,
    )

    assert len(seed_dirs) == 1
    assert summary[0].mode_source == "eon_peak"
    mode = np.loadtxt(seed_dirs[0] / "direction.dat")
    # Normalized version of the custom all-x mode.
    expected = custom_mode / np.linalg.norm(custom_mode)
    assert np.allclose(mode, expected, atol=1e-6)


def test_no_interior_peak_returns_empty(tmp_path):
    from rgpycrumbs.eon.seed_dimers import seed_dimers_from_peaks

    # Monotonic profile: no interior maximum.
    neb_con = _make_band(tmp_path, [0.0, 0.5, 1.0, 1.5, 2.0])
    seed_dirs, summary = seed_dimers_from_peaks(
        neb_con=neb_con,
        out_dir=tmp_path / "seeds",
        settings_name="s.nwi",
        socket="sock",
        prominence=0.1,
    )
    assert seed_dirs == []
    assert summary == []
