# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""modes.con from eOn's Hessian job to a metatensor TensorMap."""

from __future__ import annotations

import numpy as np
import pytest

readcon = pytest.importorskip("readcon", minversion="0.15.0")
metatensor = pytest.importorskip("metatensor")

from click.testing import CliRunner

from rgpycrumbs.eon import modes_tensormap as mod

HBAR = 0.06465415129579072
EV_TO_WAVENUMBER = 8065.543937349212


def _write_modes(path, eigenvalues, modes):
    """modes.con as eOn writes it: one frame per mode."""
    frames = []
    for k, lam in enumerate(eigenvalues):
        atoms = [
            readcon.Atom(
                symbol="H",
                x=float(i),
                y=0.0,
                z=0.0,
                mass=1.008,
                atom_id=i,
                dx=float(modes[i, 0, k]),
                dy=float(modes[i, 1, k]),
                dz=float(modes[i, 2, k]),
            )
            for i in range(modes.shape[0])
        ]
        hw = np.copysign(HBAR * np.sqrt(abs(lam)), lam)
        frames.append(
            readcon.ConFrame(
                cell=[10.0, 10.0, 10.0],
                angles=[90.0, 90.0, 90.0],
                atoms=atoms,
                metadata={
                    "frame_index": k,
                    "mode_eigenvalue": lam,
                    "hbar_omega": hw,
                    "wavenumber": hw * EV_TO_WAVENUMBER,
                },
            )
        )
    readcon.write_con(str(path), frames)


def _two_atom_modes():
    # Two atoms, six unit modes, one per Cartesian coordinate.
    modes = np.zeros((2, 3, 6))
    for k in range(6):
        modes[k // 3, k % 3, k] = 1.0
    return np.array([-0.5, 0.0, 0.25, 1.0, 2.0, 4.0]), modes


def test_modes_and_spectrum_round_trip(tmp_path):
    eigenvalues, modes = _two_atom_modes()
    _write_modes(tmp_path / "modes.con", eigenvalues, modes)
    got_modes, spectrum = mod.read_modes(tmp_path / "modes.con")
    np.testing.assert_allclose(got_modes, modes, atol=1e-6)
    np.testing.assert_allclose(spectrum[:, 0], eigenvalues)
    assert spectrum[0, 1] < 0 and spectrum[0, 2] < 0
    np.testing.assert_allclose(spectrum[3, 1], HBAR, rtol=1e-12)


def test_tensormap_blocks_carry_the_labels(tmp_path):
    eigenvalues, modes = _two_atom_modes()
    _write_modes(tmp_path / "modes.con", eigenvalues, modes)
    tensor = mod.modes_tensormap(*mod.read_modes(tmp_path / "modes.con"))
    disp = tensor.block(block=0)
    assert disp.values.shape == (2, 3, 6)
    assert disp.samples.names == ["atom"]
    assert disp.components[0].names == ["xyz"]
    assert disp.properties.names == ["mode"]
    spec = tensor.block(block=1)
    assert spec.values.shape == (6, 3)
    assert spec.samples.names == ["mode"]


def test_cli_writes_a_loadable_file(tmp_path):
    eigenvalues, modes = _two_atom_modes()
    _write_modes(tmp_path / "modes.con", eigenvalues, modes)
    out = tmp_path / "modes.mts"
    result = CliRunner().invoke(mod.main, [str(tmp_path / "modes.con"), "-o", str(out)])
    assert result.exit_code == 0, result.output
    loaded = metatensor.load(str(out))
    np.testing.assert_allclose(loaded.block(block=0).values, modes, atol=1e-6)


def test_a_frame_without_displacements_is_refused(tmp_path):
    frame = readcon.ConFrame(
        cell=[10.0, 10.0, 10.0],
        angles=[90.0, 90.0, 90.0],
        atoms=[readcon.Atom(symbol="H", x=0.0, y=0.0, z=0.0, mass=1.008)],
        metadata={"mode_eigenvalue": 1.0, "hbar_omega": 1.0, "wavenumber": 1.0},
    )
    readcon.write_con(str(tmp_path / "modes.con"), [frame])
    with pytest.raises(ValueError, match="displacements"):
        mod.read_modes(tmp_path / "modes.con")
