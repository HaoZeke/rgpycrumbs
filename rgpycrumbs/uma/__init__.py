# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""UMA AOTI package preparation for rgpot's ``UmaPot``.

rgpot ``scripts/export_uma_aoti.py`` compiles one UMA checkpoint, for
one system, into an AOTInductor ``.pt2`` that ``UmaPot`` loads. This
package keys those compiles.

An AOTI export freezes per-atom buffers at the example system's size,
and ``merge_mole`` folds the composition, charge and spin into the
weights, so a package is keyed by the exact composition, never by the
fairchem reduced composition, which accepts systems the compiled graph
then aborts on. Spin defaults to the minimal multiplicity for the
electron parity, and an explicit spin is validated against that parity.
The rest of the key (see :class:`AotiKey`) covers the model, dtype,
static-shape options and the compiler side.

.. versionadded:: 1.12.0
"""

from rgpycrumbs.uma._key import (
    KEY_SCHEMA,
    AotiKey,
    ExportEnv,
    aoti_key,
    electron_count,
    exact_counts,
    hill_formula,
    minimal_spin,
    reduced_counts,
    validate_spin,
)

__all__ = [
    "KEY_SCHEMA",
    "AotiKey",
    "ExportEnv",
    "aoti_key",
    "electron_count",
    "exact_counts",
    "hill_formula",
    "minimal_spin",
    "reduced_counts",
    "validate_spin",
]
