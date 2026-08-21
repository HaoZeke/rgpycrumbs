# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""merge_mole key and cache lookup; no fairchem / torch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rgpycrumbs.uma import (
    find_package,
    mole_key,
    prepare_uma_aoti,
    reduced_counts,
    resolve_exporter,
    write_sidecar,
)
from rgpycrumbs.uma._xyz import read_xyz, write_xyz

pytestmark = pytest.mark.pure


class TestReducedCounts:
    def test_acetylene_shares_key_with_double(self):
        c2h2 = reduced_counts([6, 6, 1, 1])
        c4h4 = reduced_counts([6, 6, 6, 6, 1, 1, 1, 1])
        assert c2h2 == ((1, 1), (6, 1))
        assert c2h2 == c4h4

    def test_ethylene_does_not_share_acetylene(self):
        assert reduced_counts([6, 6, 1, 1]) != reduced_counts([6, 6, 1, 1, 1, 1])

    def test_hcn(self):
        assert reduced_counts([6, 7, 1]) == ((1, 1), (6, 1), (7, 1))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            reduced_counts([])


class TestMoleKey:
    def test_slug_and_match(self):
        key = mole_key([6, 7, 1], charge=0, spin=1, task="omol")
        assert key.slug() == "omol-q0-s1-z1.1-z6.1-z7.1"
        assert key.matches_numbers([1, 6, 7])
        assert not key.matches_numbers([6, 6, 1, 1])

    def test_charge_splits_key(self):
        a = mole_key([6, 7, 1], charge=0, spin=1)
        b = mole_key([6, 7, 1], charge=1, spin=1)
        assert a != b


class TestCache:
    def test_find_package_hits_sidecar(self, tmp_path: Path):
        key = mole_key([6, 7, 1], charge=0, spin=1, task="omol")
        pt2 = tmp_path / f"{key.slug()}.pt2"
        pt2.write_bytes(b"pt2")
        write_sidecar(pt2, key, cutoff=6.0)
        assert find_package(tmp_path, key) == pt2

    def test_find_package_ignores_z_set_only_sidecar(self, tmp_path: Path):
        key = mole_key([6, 7, 1], charge=0, spin=1, task="omol")
        pt2 = tmp_path / "uma-s-1p1-omol-hcn.pt2"
        pt2.write_bytes(b"pt2")
        (tmp_path / "uma-s-1p1-omol-hcn.json").write_text(
            json.dumps({"task_name": "omol", "charge": 0, "spin": 1, "z_set": [1, 6, 7]})
        )
        assert find_package(tmp_path, key) is None

    def test_prepare_returns_cache_hit_without_exporter(self, tmp_path: Path):
        key = mole_key([6, 6, 1, 1], charge=0, spin=1, task="omol")
        pt2 = tmp_path / f"{key.slug()}.pt2"
        pt2.write_bytes(b"pt2")
        write_sidecar(pt2, key)
        hit = prepare_uma_aoti([6, 6, 6, 6, 1, 1, 1, 1], cache_dir=tmp_path)
        assert hit == pt2


class TestXyz:
    def test_roundtrip(self, tmp_path: Path):
        path = tmp_path / "hcn.xyz"
        numbers = [6, 7, 1]
        pos = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.1), (0.0, 0.0, -1.1)]
        write_xyz(path, numbers, pos, comment="hcn")
        got_n, got_p, comment = read_xyz(path)
        assert got_n == numbers
        assert comment == "hcn"
        assert len(got_p) == 3


class TestResolveExporter:
    def test_env_and_explicit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        script = tmp_path / "export_uma_aoti.py"
        script.write_text("# exporter\n")
        monkeypatch.delenv("RGPOT_EXPORT_UMA", raising=False)
        assert resolve_exporter(script) == script.resolve()
        monkeypatch.setenv("RGPOT_EXPORT_UMA", str(script))
        assert resolve_exporter() == script.resolve()
