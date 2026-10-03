# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""UMA AOTI package key; no fairchem or torch."""

from __future__ import annotations

import dataclasses
import json

import pytest

from rgpycrumbs.uma import (
    AotiKey,
    ExportEnv,
    aoti_key,
    exact_counts,
    hill_formula,
    minimal_spin,
    reduced_counts,
    validate_spin,
)

pytestmark = pytest.mark.pure

ENV = ExportEnv(device="cpu", target="x86_64|AVX512|", torch="2.13.0", fairchem="2.23.0")
HCN = [6, 7, 1]


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

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            exact_counts([0, 1])

    @pytest.mark.parametrize(
        ("numbers", "formula"),
        [
            ([6, 7, 1], "CHN"),
            ([6, 6, 1, 1], "C2H2"),
            ([8, 1, 1], "H2O"),
            ([26, 8], "FeO"),
        ],
    )
    def test_hill_formula(self, numbers, formula):
        assert hill_formula(exact_counts(numbers)) == formula


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

    def test_multiplicity_above_electron_count_rejected(self):
        # H2: 2 electrons admit multiplicity 1 or 3, never 5
        validate_spin([1, 1], spin=3)
        with pytest.raises(ValueError, match="needs more than 2 electrons"):
            validate_spin([1, 1], spin=5)

    def test_key_derives_doublet_for_radical(self):
        assert aoti_key([6, 6, 6, 1, 1, 1, 1, 1], ENV).spin == 2

    def test_key_rejects_parity_mismatch(self):
        with pytest.raises(ValueError, match="impossible"):
            aoti_key([6, 7, 1], ENV, spin=2)


class TestKey:
    def test_stem_names_the_system_and_ends_in_the_digest(self):
        key = aoti_key(HCN, ENV)
        assert key.stem() == f"omol-CHN-q0-s1-uma-s-1p1-cpu-{key.digest()}"
        assert len(key.digest()) == 20

    def test_digest_is_stable(self):
        # Changing this value orphans every cached package: bump KEY_SCHEMA.
        assert aoti_key(HCN, ENV).digest() == "0a7cfb8cf7b60b95cc24"

    def test_round_trip(self):
        key = aoti_key([6, 6, 1, 1], ENV, charge=1, molecular_box=25, batch_max=8)
        assert AotiKey.from_dict(json.loads(json.dumps(key.as_dict()))) == key

    def test_every_field_separates_keys(self):
        base = aoti_key(HCN, ENV)
        variants = {
            "counts": ((1, 2), (6, 2), (7, 2)),
            "charge": 2,
            "spin": 3,
            "task": "omat",
            "model": "uma-m-1p1",
            "dtype": "float64",
            "molecular_box": "25",
            "batch_max": 8,
            "device": "cuda",
            "target": "x86_64|AVX2|",
            "torch": "2.10.0",
            "fairchem": "2.21.0",
            "schema": base.schema + 1,
        }
        assert set(variants) == {f.name for f in dataclasses.fields(AotiKey)}
        keys = [base] + [dataclasses.replace(base, **{k: v}) for k, v in variants.items()]
        assert len({k.digest() for k in keys}) == len(keys)
        assert len({k.stem() for k in keys}) == len(keys)

    def test_reduced_equal_compositions_do_not_collide(self):
        a = aoti_key([6, 6, 1, 1], ENV)
        b = aoti_key([6, 6, 6, 6, 1, 1, 1, 1], ENV)
        assert a.digest() != b.digest()

    def test_atom_order_does_not_matter(self):
        assert aoti_key([1, 6, 7], ENV) == aoti_key([7, 1, 6], ENV)

    def test_equivalent_spellings_share_a_key(self):
        assert aoti_key(HCN, ENV, molecular_box=25) == aoti_key(
            HCN, ENV, molecular_box=25.0
        )
        assert aoti_key(HCN, ENV, batch_max=1) == aoti_key(HCN, ENV, batch_max=0)

    def test_negative_options_rejected(self):
        with pytest.raises(ValueError, match="molecular_box"):
            aoti_key(HCN, ENV, molecular_box=-1)
        with pytest.raises(ValueError, match="batch_max"):
            aoti_key(HCN, ENV, batch_max=-2)
