"""Parsers for ChemGP JSONL output formats.

ChemGP Rust examples produce JSONL files with method comparison data,
GP quality grids, and RFF approximation benchmarks. This module provides
structured parsing into typed containers for downstream plotting.

.. versionadded:: 1.5.0
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OptimizerTrace:
    """Single optimizer trace from a comparison JSONL.

    Attributes
    ----------
    method : str
        Optimizer name (e.g. ``"gp_minimize"``, ``"neb"``, ``"otgpd"``).
    steps : list[int]
        Step indices.
    oracle_calls : list[int]
        Cumulative oracle call counts.
    energies : list[float] | None
        Energy at each step (minimize, dimer).
    forces : list[float] | None
        Force norm at each step (dimer: ``force``, NEB: ``max_force``).
    """

    method: str
    steps: list[int] = field(default_factory=list)
    oracle_calls: list[int] = field(default_factory=list)
    energies: list[float] | None = None
    forces: list[float] | None = None


@dataclass
class ComparisonData:
    """Parsed optimizer comparison from a single JSONL file.

    Attributes
    ----------
    traces : dict[str, OptimizerTrace]
        Keyed by method name.
    summary : dict | None
        Summary record if present.
    """

    traces: dict[str, OptimizerTrace] = field(default_factory=dict)
    summary: dict[str, Any] | None = None


def parse_comparison_jsonl(path: str | Path) -> ComparisonData:
    """Parse a ChemGP optimizer comparison JSONL file.

    Handles minimize, dimer, and NEB comparison formats. Each line is a
    JSON object with a ``method`` field (or ``summary: true``).

    Parameters
    ----------
    path
        Path to the JSONL file.

    Returns
    -------
    ComparisonData
        Parsed traces keyed by method name.
    """
    data = ComparisonData()
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("summary"):
                data.summary = rec
                continue
            method = rec["method"]
            if method not in data.traces:
                data.traces[method] = OptimizerTrace(method=method)
            trace = data.traces[method]
            trace.steps.append(rec.get("step", len(trace.steps)))
            trace.oracle_calls.append(rec["oracle_calls"])
            if "energy" in rec:
                if trace.energies is None:
                    trace.energies = []
                trace.energies.append(rec["energy"])
            force_key = "force" if "force" in rec else "max_force"
            if force_key in rec:
                if trace.forces is None:
                    trace.forces = []
                trace.forces.append(rec[force_key])
    return data


@dataclass
class RFFQualityData:
    """Parsed RFF approximation quality data.

    Attributes
    ----------
    exact_energy_mae : float
        Exact GP energy MAE vs true surface.
    exact_gradient_mae : float
        Exact GP gradient MAE vs true surface.
    d_rff_values : list[int]
        RFF feature counts tested.
    energy_mae_vs_true : list[float]
        RFF energy MAE vs true surface.
    gradient_mae_vs_true : list[float]
        RFF gradient MAE vs true surface.
    energy_mae_vs_gp : list[float]
        RFF energy MAE vs exact GP.
    gradient_mae_vs_gp : list[float]
        RFF gradient MAE vs exact GP.
    """

    exact_energy_mae: float = 0.0
    exact_gradient_mae: float = 0.0
    d_rff_values: list[int] = field(default_factory=list)
    energy_mae_vs_true: list[float] = field(default_factory=list)
    gradient_mae_vs_true: list[float] = field(default_factory=list)
    energy_mae_vs_gp: list[float] = field(default_factory=list)
    gradient_mae_vs_gp: list[float] = field(default_factory=list)


def parse_rff_quality_jsonl(path: str | Path) -> RFFQualityData:
    """Parse a ChemGP RFF quality JSONL file.

    Parameters
    ----------
    path
        Path to the JSONL file.

    Returns
    -------
    RFFQualityData
        Parsed exact GP and RFF metrics.
    """
    data = RFFQualityData()
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec["type"] == "exact_gp":
                data.exact_energy_mae = rec["energy_mae"]
                data.exact_gradient_mae = rec["gradient_mae"]
            elif rec["type"] == "rff":
                data.d_rff_values.append(rec["d_rff"])
                data.energy_mae_vs_true.append(rec["energy_mae_vs_true"])
                data.gradient_mae_vs_true.append(rec["gradient_mae_vs_true"])
                data.energy_mae_vs_gp.append(rec["energy_mae_vs_gp"])
                data.gradient_mae_vs_gp.append(rec["gradient_mae_vs_gp"])
    return data


@dataclass
class GPQualityGrid:
    """GP quality grid data for a single training set size.

    Attributes
    ----------
    n_train : int
        Number of training points.
    nx : int
        Grid x resolution.
    ny : int
        Grid y resolution.
    x : list[list[float]]
        Grid x coordinates (ny x nx).
    y : list[list[float]]
        Grid y coordinates (ny x nx).
    true_e : list[list[float]]
        True energy on grid.
    gp_e : list[list[float]]
        GP predicted energy on grid.
    gp_var : list[list[float]]
        GP variance on grid.
    train_x : list[float]
        Training point x coordinates.
    train_y : list[float]
        Training point y coordinates.
    train_e : list[float]
        Training point energies.
    """

    n_train: int = 0
    nx: int = 0
    ny: int = 0
    x: list[list[float]] = field(default_factory=list)
    y: list[list[float]] = field(default_factory=list)
    true_e: list[list[float]] = field(default_factory=list)
    gp_e: list[list[float]] = field(default_factory=list)
    gp_var: list[list[float]] = field(default_factory=list)
    train_x: list[float] = field(default_factory=list)
    train_y: list[float] = field(default_factory=list)
    train_e: list[float] = field(default_factory=list)


@dataclass
class StationaryPoint:
    """A stationary point (minimum or saddle) on the PES."""

    kind: str  # "minimum" or "saddle"
    id: int
    x: float
    y: float
    energy: float


@dataclass
class GPQualityData:
    """Complete GP quality data from mb_gp_quality.jsonl.

    Attributes
    ----------
    meta : dict
        Grid metadata (nx, ny, x_min, x_max, y_min, y_max).
    stationary : list[StationaryPoint]
        Minima and saddle points.
    grids : dict[int, GPQualityGrid]
        Grid data keyed by n_train.
    """

    meta: dict[str, Any] = field(default_factory=dict)
    stationary: list[StationaryPoint] = field(default_factory=list)
    grids: dict[int, GPQualityGrid] = field(default_factory=dict)


def parse_gp_quality_jsonl(path: str | Path) -> GPQualityData:
    """Parse a ChemGP GP quality JSONL file.

    Parameters
    ----------
    path
        Path to the JSONL file (e.g. ``mb_gp_quality.jsonl``).

    Returns
    -------
    GPQualityData
        Structured grid data with metadata and stationary points.
    """
    data = GPQualityData()
    train_points = defaultdict(lambda: {"x": [], "y": [], "e": []})
    grid_records = defaultdict(list)

    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            t = rec["type"]
            if t == "grid_meta":
                data.meta = rec
            elif t in ("minimum", "saddle"):
                data.stationary.append(
                    StationaryPoint(
                        kind=t,
                        id=rec["id"],
                        x=rec["x"],
                        y=rec["y"],
                        energy=rec["energy"],
                    )
                )
            elif t == "train_point":
                n = rec["n_train"]
                train_points[n]["x"].append(rec["x"])
                train_points[n]["y"].append(rec["y"])
                train_points[n]["e"].append(rec["energy"])
            elif t == "grid":
                grid_records[rec["n_train"]].append(rec)

    nx = data.meta.get("nx", 0)
    ny = data.meta.get("ny", 0)

    for n_train, records in grid_records.items():
        grid = GPQualityGrid(n_train=n_train, nx=nx, ny=ny)
        # Initialize 2D arrays
        grid.x = [[0.0] * nx for _ in range(ny)]
        grid.y = [[0.0] * nx for _ in range(ny)]
        grid.true_e = [[0.0] * nx for _ in range(ny)]
        grid.gp_e = [[0.0] * nx for _ in range(ny)]
        grid.gp_var = [[0.0] * nx for _ in range(ny)]

        for rec in records:
            ix, iy = rec["ix"], rec["iy"]
            grid.x[iy][ix] = rec["x"]
            grid.y[iy][ix] = rec["y"]
            grid.true_e[iy][ix] = rec["true_e"]
            grid.gp_e[iy][ix] = rec["gp_e"]
            grid.gp_var[iy][ix] = rec["gp_var"]

        tp = train_points.get(n_train, {"x": [], "y": [], "e": []})
        grid.train_x = tp["x"]
        grid.train_y = tp["y"]
        grid.train_e = tp["e"]
        data.grids[n_train] = grid

    return data
