# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""UMA AOTI package keys and cache lookup.

UMA ``merge_mole`` freezes mixture-of-experts weights for one reduced
composition, charge, spin, and task. Packages are therefore reused by
that key, not by atom count or geometry.

.. versionadded:: 1.10.11
"""

from rgpycrumbs.uma._cache import find_package, write_sidecar
from rgpycrumbs.uma._key import UmaMoleKey, mole_key, reduced_counts
from rgpycrumbs.uma._prepare import prepare_uma_aoti, resolve_exporter

__all__ = [
    "UmaMoleKey",
    "find_package",
    "mole_key",
    "prepare_uma_aoti",
    "reduced_counts",
    "resolve_exporter",
    "write_sidecar",
]
