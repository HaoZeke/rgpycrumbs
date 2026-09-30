# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Bead centroid and ring-polymer spread from PIMD trajectories."""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from rgpycrumbs.eon import pimd_centroid as mod

BOHR = 0.529177210544


def _ring(p, n_frames=3, radius=0.4):
    """P beads on a circle in the xy plane around a drifting centroid.

    Two atoms, atom 1 on a ring of twice the radius. Returns the bead
    positions ``(P, F, 2, 3)`` in Angstrom and the centroids ``(F, 2, 3)``.
    """
    centres = np.zeros((n_frames, 2, 3))
    for f in range(n_frames):
        centres[f, 0] = [1.0 + 0.1 * f, 2.0, 3.0]
        centres[f, 1] = [4.0, 5.0 + 0.2 * f, 6.0]
    beads = np.zeros((p, n_frames, 2, 3))
    for k in range(p):
        phi = 2 * np.pi * k / p
        for a, r in enumerate((radius, 2 * radius)):
            beads[k, :, a] = centres[:, a] + r * np.array([np.cos(phi), np.sin(phi), 0])
    return beads, centres


def _write_cpmd(path, frames_bohr, steps=None):
    """TRAJECTORY text: step, x, y, z, vx, vy, vz per atom."""
    lines = []
    for f, frame in enumerate(frames_bohr):
        step = f + 1 if steps is None else steps[f]
        lines.extend(
            f"{step:8d} " + " ".join(f"{v:.10f}" for v in (*row, 0.1, 0.2, 0.3))
            for row in frame
        )
    path.write_text("\n".join(lines) + "\n")


def _write_xyz(path, frames, symbols=("H", "O")):
    out = []
    for frame in frames:
        out.append(str(len(frame)))
        out.append("frame")
        out.extend(
            f"{s} " + " ".join(f"{v:.10f}" for v in row)
            for s, row in zip(symbols, frame, strict=True)
        )
    path.write_text("\n".join(out) + "\n")


@pytest.mark.parametrize("p", [3, 4, 8])
def test_ring_gives_centroid_and_analytic_spread(p):
    beads, centres = _ring(p)
    centroid, spread = mod.centroid_spread(beads)
    np.testing.assert_allclose(centroid, centres, atol=1e-12)
    # rms of r cos(phi) over P > 2 equally spaced beads is r / sqrt(2).
    np.testing.assert_allclose(spread[..., 0], [[0.4, 0.8]] * 3 / np.sqrt(2))
    np.testing.assert_allclose(spread[..., 1], spread[..., 0])
    np.testing.assert_allclose(spread[..., 2], 0.0, atol=1e-12)


def test_trajectory_converts_bohr_to_angstrom(tmp_path):
    beads, _ = _ring(4)
    _write_cpmd(tmp_path / "TRAJECTORY", beads[0] / BOHR)
    got, symbols = mod.read_trajectory(tmp_path / "TRAJECTORY")
    assert symbols is None
    assert got.shape == (3, 2, 3)
    np.testing.assert_allclose(got, beads[0], atol=1e-8)


def test_xyz_positions_are_angstrom(tmp_path):
    beads, _ = _ring(4)
    _write_xyz(tmp_path / "TRAJEC.xyz", beads[0])
    got, symbols = mod.read_trajectory(tmp_path / "TRAJEC.xyz")
    assert symbols == ["H", "O"]
    np.testing.assert_allclose(got, beads[0], atol=1e-8)


def test_single_file_and_multi_file_layouts_agree(tmp_path):
    p = 4
    beads, _ = _ring(p)
    files = []
    for k in range(p):
        f = tmp_path / f"TRAJECTORY_{k}"
        _write_cpmd(f, beads[k] / BOHR)
        files.append(f)
    # Replica-major: replica 0's atoms, then replica 1's, in each frame.
    merged = np.concatenate(list(beads), axis=1)
    _write_cpmd(tmp_path / "TRAJECTORY", merged / BOHR)
    multi, _ = mod.load_replicas(files)
    single, _ = mod.load_replicas([tmp_path / "TRAJECTORY"], replicas=p)
    np.testing.assert_allclose(single, multi)
    a = mod.collapse(multi)
    b = mod.collapse(single)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    assert a[2] == b[2]


def test_xyz_single_file_splits_species_too(tmp_path):
    beads, _ = _ring(3)
    merged = np.concatenate(list(beads), axis=1)
    _write_xyz(tmp_path / "all.xyz", merged, symbols=("H", "O") * 3)
    got, symbols = mod.load_replicas([tmp_path / "all.xyz"], replicas=3)
    assert symbols == ["H", "O"]
    np.testing.assert_allclose(got, beads, atol=1e-8)


def test_every_keeps_each_kth_frame_with_original_index():
    beads, centres = _ring(4, n_frames=7)
    pos, spread, md = mod.collapse(beads, every=3)
    assert pos.shape[0] == 3
    np.testing.assert_allclose(pos, centres[::3], atol=1e-12)
    assert [m["frame_index"] for m in md] == [0, 3, 6]
    assert all(m["replicas"] == 4 for m in md)
    np.testing.assert_allclose(md[0]["spread_max"], 0.8 / np.sqrt(2))


def test_time_average_combines_both_spreads():
    beads, centres = _ring(4, n_frames=5)
    pos, spread, md = mod.collapse(beads, average=True)
    assert pos.shape == (1, 2, 3) and len(md) == 1
    np.testing.assert_allclose(pos[0], centres.mean(axis=0), atol=1e-12)
    thermal = np.sqrt(((centres - centres.mean(axis=0)) ** 2).mean(axis=0))
    imag_xy = np.array([0.4, 0.8]) / np.sqrt(2)
    want_x = np.sqrt(imag_xy**2 + thermal[:, 0] ** 2)
    np.testing.assert_allclose(spread[0, :, 0], want_x)
    m = md[0]
    assert m["frames_averaged"] == 5
    np.testing.assert_allclose(
        m["spread_imaginary_time_rms"], np.sqrt((2 * imag_xy**2).sum() / 6)
    )
    np.testing.assert_allclose(m["spread_thermal_rms"], np.sqrt((thermal**2).mean()))


def test_frame_count_mismatch_is_refused(tmp_path):
    beads, _ = _ring(3)
    _write_cpmd(tmp_path / "a", beads[0] / BOHR)
    _write_cpmd(tmp_path / "b", beads[1][:2] / BOHR)
    with pytest.raises(ValueError, match="frame or atom count"):
        mod.load_replicas([tmp_path / "a", tmp_path / "b"])


def test_atom_count_mismatch_is_refused(tmp_path):
    beads, _ = _ring(3)
    _write_cpmd(tmp_path / "a", beads[0] / BOHR)
    _write_cpmd(tmp_path / "b", beads[1][:, :1] / BOHR)
    with pytest.raises(ValueError, match="frame or atom count"):
        mod.load_replicas([tmp_path / "a", tmp_path / "b"])


def test_replicas_that_do_not_divide_the_atoms_are_refused(tmp_path):
    beads, _ = _ring(3)
    _write_cpmd(tmp_path / "a", beads[0] / BOHR)
    with pytest.raises(ValueError, match="do not split"):
        mod.load_replicas([tmp_path / "a"], replicas=3)


def test_cli_maps_a_mismatch_to_a_usage_error(tmp_path):
    beads, _ = _ring(3)
    _write_cpmd(tmp_path / "a", beads[0] / BOHR)
    _write_cpmd(tmp_path / "b", beads[1][:2] / BOHR)
    result = CliRunner().invoke(
        mod.main,
        [str(tmp_path / "a"), str(tmp_path / "b"), "-o", str(tmp_path / "o.con")],
    )
    assert result.exit_code == 2
    assert "frame or atom count" in result.output
    assert "Traceback" not in result.output


def test_cli_missing_file_is_a_click_error(tmp_path):
    result = CliRunner().invoke(mod.main, [str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def _readcon():
    return pytest.importorskip("readcon", minversion="0.16.0")


def test_con_carries_centroid_and_spread(tmp_path):
    readcon = _readcon()
    beads, centres = _ring(4)
    files = []
    for k in range(4):
        f = tmp_path / f"TRAJECTORY_{k}"
        _write_cpmd(f, beads[k] / BOHR)
        files.append(str(f))
    reference = tmp_path / "reactant.con"
    readcon.write_con(
        str(reference),
        [
            readcon.ConFrame(
                cell=[10.0, 11.0, 12.0],
                angles=[90.0, 90.0, 90.0],
                atoms=[
                    readcon.Atom(
                        symbol="H", x=0.0, y=0.0, z=0.0, mass=1.008, fixed=[True] * 3
                    ),
                    readcon.Atom(symbol="O", x=1.0, y=0.0, z=0.0, mass=15.999, atom_id=1),
                ],
            )
        ],
    )
    out = tmp_path / "centroid.con"
    result = CliRunner().invoke(
        mod.main, [*files, "--reference", str(reference), "--every", "2", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    frames = readcon.read_con(str(out))
    assert len(frames) == 2
    np.testing.assert_allclose(frames[1].coords_array(), centres[2], atol=1e-6)
    np.testing.assert_allclose(
        frames[0].spread[:, 0], np.array([0.4, 0.8]) / np.sqrt(2), atol=1e-6
    )
    assert list(frames[0].cell) == [10.0, 11.0, 12.0]
    assert frames[0].atoms[0].is_fixed
    assert frames[0].metadata["replicas"] == 4
    assert frames[1].metadata["frame_index"] == 2


def test_con_time_average_metadata(tmp_path):
    readcon = _readcon()
    beads, _ = _ring(4)
    _write_xyz(tmp_path / "all.xyz", np.concatenate(list(beads), axis=1), ("H", "O") * 4)
    out = tmp_path / "avg.con"
    result = CliRunner().invoke(
        mod.main,
        [
            str(tmp_path / "all.xyz"),
            "--replicas",
            "4",
            "--cell",
            "9",
            "9",
            "9",
            "--time-average",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    (frame,) = readcon.read_con(str(out))
    assert {"spread_imaginary_time_rms", "spread_thermal_rms", "spread_max"} <= set(
        frame.metadata
    )
