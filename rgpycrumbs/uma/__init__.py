# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""UMA AOTI package keys and cache lookup.

An AOTI export freezes per-atom buffers at the example system's size,
so packages are keyed and reused by exact composition, charge, spin
multiplicity, and task -- never by the fairchem ``merge_mole`` reduced
composition, which accepts systems the compiled graph then aborts on.
Spin defaults to the minimal multiplicity for the electron parity, and
an explicit spin is validated against that parity.

.. versionadded:: 1.10.11
"""

from rgpycrumbs.uma._cache import find_package, write_sidecar
from rgpycrumbs.uma._key import (
    UmaMoleKey,
    electron_count,
    exact_counts,
    minimal_spin,
    mole_key,
    reduced_counts,
    validate_spin,
)
from rgpycrumbs.uma._prepare import prepare_uma_aoti, resolve_exporter

__all__ = [
    "UmaMoleKey",
    "electron_count",
    "exact_counts",
    "find_package",
    "minimal_spin",
    "mole_key",
    "prepare_uma_aoti",
    "reduced_counts",
    "resolve_exporter",
    "validate_spin",
    "write_sidecar",
]
