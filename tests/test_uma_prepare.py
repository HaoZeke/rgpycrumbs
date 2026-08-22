# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""merge_mole key and cache lookup; no fairchem / torch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rgpycrumbs.uma import (
    exact_counts,
    find_package,
    minimal_spin,
    mole_key,
    prepare_uma_aoti,
    reduced_counts,
    resolve_exporter,
    validate_spin,
    write_sidecar,
)
pytestmark = pytest.mark.pure


class TestCounts:
    def test_reduced_is_model_side_only(self):
        c2h2 = reduced_counts([6, 6, 1, 1])
        c4h4 = reduced_counts([6, 6, 6, 6, 1, 1, 1, 1])
        assert c2h2 == ((1, 1), (6, 1))
        assert c2h2 == c4h4

    def test_exact_separates_acetylene_from_double(self):
        assert exact_counts([6, 6, 1, 1]) == ((1, 2), (6, 2))
        assert exact_counts([6, 6, 1, 1]) != exact_counts([6, 6, 6, 6, 1, 1, 1, 1])

    def test_hcn(self):
        assert exact_counts([6, 7, 1]) == ((1, 1), (6, 1), (7, 1))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            exact_counts([])


class TestSpinParity:
    def test_even_electrons_default_singlet(self):
        assert minimal_spin([6, 7, 1]) == 1

    def test_odd_electrons_default_doublet(self):
        # cyclopropyl C3H5: 23 electrons
        assert minimal_spin([6, 6, 6, 1, 1, 1, 1, 1]) == 2

    def test_charge_flips_parity(self):
        assert minimal_spin([6, 6, 6, 1, 1, 1, 1, 1], charge=1) == 1

    def test_singlet_radical_rejected(self):
        with pytest.raises(ValueError, match="impossible for 23 electrons"):
            validate_spin([6, 6, 6, 1, 1, 1, 1, 1], spin=1)

    def test_key_derives_doublet_for_radical(self):
        key = mole_key([6, 6, 6, 1, 1, 1, 1, 1])
        assert key.spin == 2

    def test_key_rejects_parity_mismatch(self):
        with pytest.raises(ValueError, match="impossible"):
            mole_key([6, 7, 1], spin=2)


class TestMoleKey:
    def test_slug_and_match(self):
        key = mole_key([6, 7, 1], charge=0, spin=1, task="omol")
        assert key.slug() == "omol-q0-s1-z1.1-z6.1-z7.1"
        assert key.matches_numbers([1, 6, 7])
        assert not key.matches_numbers([6, 6, 1, 1])

    def test_exact_key_never_matches_multiple(self):
        key = mole_key([6, 6, 1, 1], charge=0, spin=1, task="omol")
        assert not key.matches_numbers([6, 6, 6, 6, 1, 1, 1, 1])

    def test_charge_splits_key(self):
        a = mole_key([6, 7, 1], charge=0)
        b = mole_key([6, 7, 1], charge=1)
        assert a != b
        assert a.spin == 1
        assert b.spin == 2


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
        hit = prepare_uma_aoti([6, 6, 1, 1], cache_dir=tmp_path)
        assert hit == pt2

    def test_prepare_misses_on_different_atom_count(self, tmp_path: Path):
        key = mole_key([6, 6, 1, 1], charge=0, spin=1, task="omol")
        pt2 = tmp_path / f"{key.slug()}.pt2"
        pt2.write_bytes(b"pt2")
        write_sidecar(pt2, key)
        with pytest.raises(ValueError, match="cache miss"):
            prepare_uma_aoti([6, 6, 6, 6, 1, 1, 1, 1], cache_dir=tmp_path)


class TestResolveExporter:
    def test_env_and_explicit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        script = tmp_path / "export_uma_aoti.py"
        script.write_text("# exporter\n")
        monkeypatch.delenv("RGPOT_EXPORT_UMA", raising=False)
        assert resolve_exporter(script) == script.resolve()
        monkeypatch.setenv("RGPOT_EXPORT_UMA", str(script))
        assert resolve_exporter() == script.resolve()
