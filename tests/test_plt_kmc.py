# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for rgpycrumbs.eon.plt_kmc (aKMC N2-ejection visualization)."""

import numpy as np
import pytest

from tests._optional_imports import has_module_spec

pytestmark = pytest.mark.skipif(
    not all(
        has_module_spec(m)
        for m in ("readcon", "ase", "matplotlib", "scipy", "chemparseplot")
    ),
    reason="plt_kmc needs readcon + ase + matplotlib + scipy + chemparseplot",
)

_DYNAMICS = """\
 step-number   reactant-id    process-id    product-id     step-time    total-time       barrier          rate        energy
-----------------------------------------------------------------------------------------------------------------------------
           0             0             3             1  1.000000e-09  1.000000e-09      0.230000  6.000000e+08  -1206.800000
           1             1             0             2  2.000000e-09  3.000000e-09      0.210000  4.000000e+08  -1206.900000
"""


def _write_state(states_dir, sid, n2_offset):
    """State with a Si3 core at origin and one N2 displaced by *n2_offset*."""
    from ase import Atoms
    from readcon import ConFrame, write_con

    core = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
    n2 = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]) + n2_offset
    atoms = Atoms(
        "Si3N2",
        positions=np.vstack([core, n2]),
        cell=[30, 30, 30],
    )
    sdir = states_dir / str(sid)
    sdir.mkdir(parents=True)
    write_con(str(sdir / "reactant.con"), [ConFrame.from_ase(atoms)])


def _build_akmc(tmp_path):
    akmc = tmp_path / "akmc"
    states = akmc / "states"
    (akmc).mkdir()
    (akmc / "dynamics.txt").write_text(_DYNAMICS)
    # state 0: N2 near core (not ejected); states 1,2: N2 far away (ejected).
    _write_state(states, 0, np.array([1.0, 0.0, 0.0]))
    _write_state(states, 1, np.array([10.0, 0.0, 0.0]))
    _write_state(states, 2, np.array([12.0, 0.0, 0.0]))
    return akmc


def test_parse_dynamics_and_sequence(tmp_path):
    from rgpycrumbs.eon.plt_kmc import parse_dynamics, visited_state_sequence

    akmc = _build_akmc(tmp_path)
    steps = parse_dynamics(akmc / "dynamics.txt")
    assert len(steps) == 2
    assert steps[0].reactant_id == 0
    assert steps[0].product_id == 1
    assert steps[1].total_time == pytest.approx(3.0e-9)
    assert visited_state_sequence(steps) == [0, 1, 2]


def test_count_ejected_n2():
    from ase import Atoms

    from rgpycrumbs.eon.plt_kmc import count_ejected_n2

    near = Atoms(
        "Si3N2",
        positions=np.array(
            [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.0, 0, 0], [2.1, 0, 0]]
        ),
        cell=[30, 30, 30],
    )
    assert count_ejected_n2(near, core_threshold=4.0) == 0

    far = Atoms(
        "Si3N2",
        positions=np.array(
            [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [10.0, 0, 0], [11.1, 0, 0]]
        ),
        cell=[30, 30, 30],
    )
    assert count_ejected_n2(far, core_threshold=4.0) == 1


def test_collect_timeline_and_plot(tmp_path):
    from rgpycrumbs.eon.plt_kmc import collect_kmc_n2_timeline, plot_kmc_timeline

    akmc = _build_akmc(tmp_path)
    timeline = collect_kmc_n2_timeline(akmc)

    assert timeline.state_ids == [0, 1, 2]
    # state 0 -> 0 ejected, states 1 and 2 -> 1 ejected each (cumulative count).
    assert timeline.cumulative_n2 == [0, 1, 1]
    assert len(timeline.times) == 3
    assert len(timeline.energies) == 3

    out = tmp_path / "kmc.png"
    written = plot_kmc_timeline(timeline, out)
    assert written.is_file()
    assert written.stat().st_size > 0


def test_robust_to_missing_state(tmp_path):
    from rgpycrumbs.eon.plt_kmc import collect_kmc_n2_timeline

    akmc = _build_akmc(tmp_path)
    # Remove state 2's reactant.con to simulate a partial aKMC dir.
    (akmc / "states" / "2" / "reactant.con").unlink()
    timeline = collect_kmc_n2_timeline(akmc)
    # state 2 reuses the previous (state 1) count of 1.
    assert timeline.cumulative_n2 == [0, 1, 1]
