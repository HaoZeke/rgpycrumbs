# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Object-aware plot() dispatch: types and ConFrame sequences (no path sniff)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._optional_imports import has_module_spec

pytestmark = pytest.mark.pure


def _fake_frame(energy: float, idx: int):
    """Duck-typed ConFrame stand-in (fields plot adapters need)."""

    class _Fr:
        def __init__(self):
            self.energy = energy
            self.frame_index = idx
            self.metadata = {
                "energy": energy,
                "frame_index": idx,
                "step_size": 0.1,
                "convergence": 0.01,
                "reaction_coordinate": float(idx),
                "relative_energy": energy,
                "parallel_force": 0.0,
                "delta_e": energy,
                "eigenvalue": -1.0,
                "torque": 0.0,
                "angle": 0.0,
                "rotations": 0,
            }
            self.neb_bead = idx

        def to_ase(self):
            pytest.importorskip("ase")
            from ase import Atoms

            a = Atoms("H", positions=[[0.0, 0.0, float(idx)]], cell=[5, 5, 5])
            a.info["energy"] = float(self.energy)
            return a

    return _Fr()


def test_adapt_frame_sequence_requires_kind():
    from rgpycrumbs.eon.plot_dispatch import adapt_plot_source

    frames = [_fake_frame(1.0, 0), _fake_frame(1.1, 1)]
    with pytest.raises(TypeError, match="kind="):
        adapt_plot_source(frames)


def test_adapt_frame_sequence_neb_min_saddle():
    from rgpycrumbs.eon.plot_dispatch import adapt_plot_source

    frames = [_fake_frame(1.0, 0), _fake_frame(1.2, 1), _fake_frame(1.1, 2)]
    cmd, payload = adapt_plot_source(frames, kind="neb")
    assert cmd == "neb"
    assert len(payload["frames"]) == 3

    cmd, payload = adapt_plot_source(frames, kind="min")
    assert cmd == "min"
    assert "frames" in payload

    cmd, payload = adapt_plot_source(frames, kind="saddle")
    assert cmd == "saddle"


def test_adapt_neb_like_path_frames():
    from rgpycrumbs.eon.plot_dispatch import adapt_plot_source

    frames = [_fake_frame(0.0, i) for i in range(4)]

    class FakeNEB:
        __module__ = "pyeonclient._core"

        def compute(self):
            return 0

        def path_frames(self):
            return frames

    cmd, payload = adapt_plot_source(FakeNEB())
    assert cmd == "neb"
    assert len(payload["frames"]) == 4


def test_adapt_matter_movie_frames():
    from rgpycrumbs.eon.plot_dispatch import adapt_plot_source

    frames = [_fake_frame(-1.0, i) for i in range(3)]

    class FakeMatter:
        __module__ = "pyeonclient._core"

        def relax(self, **kw):
            return self, True

        def movie_frames(self):
            return frames

    cmd, payload = adapt_plot_source(FakeMatter())
    assert cmd == "min"
    assert len(payload["frames"]) == 3


def test_adapt_saddle_climb_frames():
    from rgpycrumbs.eon.plot_dispatch import adapt_plot_source

    frames = [_fake_frame(0.5, i) for i in range(2)]

    class FakeSS:
        __module__ = "pyeonclient._core"

        def run_retain_frames(self, **kw):
            return 0

        def climb_frames(self):
            return frames

    cmd, payload = adapt_plot_source(FakeSS())
    assert cmd == "saddle"
    assert len(payload["frames"]) == 2


def test_adapt_rejects_path_without_type():
    from pathlib import Path

    from rgpycrumbs.eon.plot_dispatch import adapt_plot_source

    with pytest.raises(TypeError, match="does not support"):
        adapt_plot_source(Path("/tmp/mixed_job_dir"))


def test_plot_merges_settings_and_calls_runner(monkeypatch, tmp_path):
    """plot() uses run_plot merge + command runner (no durable con required)."""
    from rgpycrumbs.eon import plot_dispatch

    seen: list[tuple[str, dict]] = []

    def fake_run_plot(command, runner, *, config=None, **overrides):
        settings = {"command": command, **overrides}
        seen.append((command, settings))
        # call runner with minimal dict so we exercise payload wiring
        return runner({**settings, "plot_type": "profile", "verbose": False})

    monkeypatch.setattr(plot_dispatch, "run_plot", fake_run_plot)

    # stub runners to avoid heavy plot stack
    def neb_runner(settings):
        assert "frames" in settings
        return tmp_path / "neb.pdf"

    def min_runner(settings):
        assert "trajectory" in settings or "frames" in settings
        return tmp_path / "min.pdf"

    monkeypatch.setattr(
        plot_dispatch,
        "_runner_for",
        lambda command: {"neb": neb_runner, "min": min_runner, "saddle": neb_runner}[
            command
        ],
    )

    frames = [_fake_frame(1.0, i) for i in range(3)]
    out = plot_dispatch.plot(frames, kind="neb", plot_type="profile")
    assert out == tmp_path / "neb.pdf"
    assert seen[0][0] == "neb"
    assert "frames" in seen[0][1]


@pytest.mark.skipif(
    not all(has_module_spec(m) for m in ("readcon", "ase", "polars", "chemparseplot")),
    reason="frame_series needs readcon+chemparseplot",
)
def test_plot_min_frames_builds_trajectory(monkeypatch, tmp_path):
    from rgpycrumbs.eon import plot_dispatch

    captured = {}

    def min_runner(settings):
        captured.update(settings)
        assert "trajectory" in settings
        traj = settings["trajectory"]
        assert len(traj.atoms_list) == 3
        assert traj.dat_df.height == 3
        return tmp_path / "out.pdf"

    monkeypatch.setattr(
        plot_dispatch,
        "_runner_for",
        lambda command: min_runner if command == "min" else (_ for _ in ()).throw(
            AssertionError(command)
        ),
    )
    # keep real run_plot
    frames = [_fake_frame(-2.0 + 0.1 * i, i) for i in range(3)]
    # need real metadata for min table - fake frames have step_size etc.
    out = plot_dispatch.plot(
        frames, kind="min", plot_type="profile", output=tmp_path / "out.pdf"
    )
    assert out == tmp_path / "out.pdf"
    assert "trajectory" in captured


@pytest.mark.skipif(
    not has_module_spec("chemparseplot"),
    reason="export needs package",
)
def test_public_plot_export():
    from rgpycrumbs.eon import adapt_plot_source, plot

    assert callable(plot)
    assert callable(adapt_plot_source)
