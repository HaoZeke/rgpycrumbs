# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Typed records for NEB / saddle data.

Canonical definitions live in ``chemparseplot.basetypes``. This module
re-exports them for backward compatibility. Prefer importing from
chemparseplot in new parse/plot library code.

When chemparseplot is not installed (hub-only environment), a local
fallback copy is used so pure hub tests still import.
"""

from __future__ import annotations

try:
    from chemparseplot.basetypes import (
        DimerOpt,
        MolGeom,
        SaddleMeasure,
        SpinID,
        nebiter,
        nebpath,
    )
except ImportError:  # pragma: no cover - hub-only fallback
    import datetime
    from dataclasses import dataclass, field

    import numpy as np

    @dataclass(frozen=True, slots=True)
    class nebpath:
        norm_dist: float
        arc_dist: float
        energy: float

    @dataclass(frozen=True, slots=True)
    class nebiter:
        iteration: int
        nebpath: nebpath

    @dataclass
    class DimerOpt:
        saddle: str = "dimer"
        rot: str = "lbfgs"
        trans: str = "lbfgs"

    @dataclass
    class SpinID:
        mol_id: int
        spin: str

    @dataclass
    class MolGeom:
        pos: np.ndarray
        energy: float
        forces: np.ndarray

    @dataclass
    class SaddleMeasure:
        pes_calls: int = 0
        iter_steps: int = 0
        tot_time: float = field(
            default_factory=lambda: datetime.timedelta(0).total_seconds()
        )
        saddle_energy: float = np.nan
        saddle_fmax: float = np.nan
        success: bool = False
        method: str = "not run"
        dimer_rot: str = "n/a"
        dimer_trans: str = "n/a"
        init_energy: float = np.nan
        barrier: float = np.nan
        mol_id: int = np.nan
        spin: str = "unknown"
        scf: float = np.nan
        termination_status: str = "not set"

__all__ = [
    "DimerOpt",
    "MolGeom",
    "SaddleMeasure",
    "SpinID",
    "nebiter",
    "nebpath",
]
