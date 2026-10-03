# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""UMA AOTI package preparation for rgpot's ``UmaPot``.

rgpot ``scripts/export_uma_aoti.py`` compiles one UMA checkpoint, for
one system, into an AOTInductor ``.pt2`` that ``UmaPot`` loads. This
package keys those compiles, caches them, and runs the exporter only on
a miss.

An AOTI export freezes per-atom buffers at the example system's size,
and ``merge_mole`` folds the composition, charge and spin into the
weights, so a package is keyed by the exact composition, never by the
fairchem reduced composition, which accepts systems the compiled graph
then aborts on. Spin defaults to the minimal multiplicity for the
electron parity, and an explicit spin is validated against that parity.
The rest of the key (see :class:`AotiKey`) covers the model, dtype,
static-shape options and the compiler side.

The library is standard library only; torch and fairchem are needed by
the exporter, which runs in the interpreter of the
``rgpycrumbs uma prepare-aoti`` script.

.. versionadded:: 1.12.0
"""

from rgpycrumbs.uma._cache import (
    CacheEntry,
    default_cache_dir,
    embedded_metadata,
    embedded_mismatches,
    entries,
    find_entry,
    lookup,
    package_path,
    remove,
    sidecar_path,
)
from rgpycrumbs.uma._env import probe_env
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
from rgpycrumbs.uma._prepare import Prepared, prepare_uma_aoti, resolve_exporter

__all__ = [
    "KEY_SCHEMA",
    "AotiKey",
    "CacheEntry",
    "ExportEnv",
    "Prepared",
    "aoti_key",
    "default_cache_dir",
    "electron_count",
    "embedded_metadata",
    "embedded_mismatches",
    "entries",
    "exact_counts",
    "find_entry",
    "hill_formula",
    "lookup",
    "minimal_spin",
    "package_path",
    "prepare_uma_aoti",
    "probe_env",
    "reduced_counts",
    "remove",
    "resolve_exporter",
    "sidecar_path",
    "validate_spin",
]
