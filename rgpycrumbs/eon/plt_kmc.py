#!/usr/bin/env python3
"""Visualize an eOn aKMC trajectory: N2 ejection vs simulation time.

.. versionadded:: 1.8.0

Reads an eOn aKMC output directory (``dynamics.txt`` + per-state
``states/<id>/reactant.con``) and plots, against cumulative KMC simulation
time:

1. the cumulative number of N2 molecules ejected from the Si core, and
2. the system energy and visited-state index.

An N2 is detected as a bonded N-N pair whose midpoint lies further than a
threshold (default 4 Angstrom) from the Si-core center of mass. The reader is
robust to partial aKMC directories (missing states are skipped).
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "rich",
#   "ase",
#   "readcon>=0.7.0",
#   "chemparseplot[neb,plot]>=1.9.17,<2",
#   "rgpycrumbs>=1.10.4",
# ]
# ///

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

try:
    from rgpycrumbs._aux import warn_on_direct_script_import
except ImportError:  # pragma: no cover - direct script execution without package root
    warn_on_direct_script_import = None

if warn_on_direct_script_import is not None:
    warn_on_direct_script_import(__name__, "rgpycrumbs eon plt-kmc")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DynamicsStep:
    """One accepted aKMC transition from ``dynamics.txt``.

    .. versionadded:: 1.8.0
    """

    step: int
    reactant_id: int
    process_id: int
    product_id: int
    step_time: float
    total_time: float
    barrier: float
    rate: float
    energy: float


# Column-name -> DynamicsStep field, tolerant of eOn header spellings.
_COLUMN_ALIASES = {
    "step-number": "step",
    "step": "step",
    "reactant-id": "reactant_id",
    "process-id": "process_id",
    "product-id": "product_id",
    "step-time": "step_time",
    "total-time": "total_time",
    "barrier": "barrier",
    "rate": "rate",
    "energy": "energy",
}
_DEFAULT_ORDER = [
    "step",
    "reactant_id",
    "process_id",
    "product_id",
    "step_time",
    "total_time",
    "barrier",
    "rate",
    "energy",
]


def parse_dynamics(dynamics_file) -> list[DynamicsStep]:
    """Parse an eOn aKMC ``dynamics.txt`` into ordered :class:`DynamicsStep`.

    Uses the header row to map columns when present; otherwise assumes the
    canonical eOn column order. Separator and comment lines are skipped.

    .. versionadded:: 1.8.0
    """
    dynamics_file = Path(dynamics_file)
    fields = _DEFAULT_ORDER
    steps: list[DynamicsStep] = []
    for raw in dynamics_file.read_text().splitlines():
        line = raw.strip()
        if not line or set(line) <= {"-", "="}:
            continue
        tokens = line.split()
        lower = [t.lower() for t in tokens]
        if any(tok in _COLUMN_ALIASES for tok in lower):
            mapped = [_COLUMN_ALIASES.get(tok) for tok in lower]
            if any(m is not None for m in mapped):
                fields = mapped
                continue
        values: dict[str, float] = {}
        for field, tok in zip(fields, tokens):
            if field is None:
                continue
            try:
                values[field] = float(tok)
            except ValueError:
                values = {}
                break
        if not values:
            continue
        steps.append(
            DynamicsStep(
                step=int(values.get("step", len(steps))),
                reactant_id=int(values.get("reactant_id", -1)),
                process_id=int(values.get("process_id", -1)),
                product_id=int(values.get("product_id", -1)),
                step_time=values.get("step_time", float("nan")),
                total_time=values.get("total_time", float("nan")),
                barrier=values.get("barrier", float("nan")),
                rate=values.get("rate", float("nan")),
                energy=values.get("energy", float("nan")),
            )
        )
    return steps


def visited_state_sequence(steps: list[DynamicsStep]) -> list[int]:
    """Ordered list of visited state ids: initial reactant then each product.

    .. versionadded:: 1.8.0
    """
    if not steps:
        return []
    seq = [steps[0].reactant_id]
    seq.extend(s.product_id for s in steps)
    return seq


def count_ejected_n2(
    atoms,
    core_threshold: float = 4.0,
    nn_cutoff: float = 1.6,
) -> int:
    """Count N2 molecules ejected from the Si core in an ASE ``Atoms``.

    An N2 is a bonded N-N pair (separation < ``nn_cutoff``) whose midpoint lies
    further than ``core_threshold`` from the Si-atom center of mass. N atoms are
    greedily paired nearest-first. Falls back to the all-atom COM as the core
    reference when no Si atoms are present.

    .. versionadded:: 1.8.0
    """
    symbols = np.asarray(atoms.get_chemical_symbols())
    positions = np.asarray(atoms.get_positions(), dtype=float)

    si_mask = symbols == "Si"
    if si_mask.any():
        core_com = positions[si_mask].mean(axis=0)
    elif len(positions):
        core_com = positions.mean(axis=0)
    else:
        return 0

    n_idx = np.flatnonzero(symbols == "N")
    if n_idx.size < 2:
        return 0

    n_pos = positions[n_idx]
    # Greedy nearest-neighbour pairing of N atoms.
    diff = n_pos[:, None, :] - n_pos[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)

    pairs: list[tuple[int, int]] = []
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    used: set[int] = set()
    for i, j in order:
        if dist[i, j] > nn_cutoff:
            break
        if i in used or j in used:
            continue
        pairs.append((int(i), int(j)))
        used.update((int(i), int(j)))

    ejected = 0
    for i, j in pairs:
        midpoint = 0.5 * (n_pos[i] + n_pos[j])
        if np.linalg.norm(midpoint - core_com) > core_threshold:
            ejected += 1
    return ejected


@dataclass
class KmcTimeline:
    """Cumulative N2 ejection and energy vs time across visited states.

    .. versionadded:: 1.8.0
    """

    times: list[float]
    cumulative_n2: list[int]
    energies: list[float]
    state_ids: list[int]


def collect_kmc_n2_timeline(
    akmc_dir,
    core_threshold: float = 4.0,
    nn_cutoff: float = 1.6,
) -> KmcTimeline:
    """Build the N2-ejection timeline for an eOn aKMC directory.

    Robust to partial directories: states missing ``reactant.con`` reuse the
    previous count. Requires ``readcon`` and ``ase``.

    .. versionadded:: 1.8.0
    """
    from readcon import read_con

    akmc_dir = Path(akmc_dir)
    steps = parse_dynamics(akmc_dir / "dynamics.txt")
    sequence = visited_state_sequence(steps)
    # Energy/time aligned with the visited-state sequence (initial state first).
    times = [0.0] + [s.total_time for s in steps]
    energies = [steps[0].energy if steps else float("nan")] + [
        s.energy for s in steps
    ]

    counts: dict[int, int] = {}
    last_count = 0
    cumulative: list[int] = []
    for state_id in sequence:
        if state_id in counts:
            last_count = counts[state_id]
        else:
            con = akmc_dir / "states" / str(state_id) / "reactant.con"
            if con.is_file():
                try:
                    frame = read_con(str(con))[0]
                    last_count = count_ejected_n2(
                        frame.to_ase(), core_threshold, nn_cutoff
                    )
                except Exception as exc:  # noqa: BLE001 - robust to bad/partial files
                    log.warning("state %s: could not read %s (%s)", state_id, con, exc)
                else:
                    counts[state_id] = last_count
            else:
                log.warning("state %s: missing %s; reusing previous count", state_id, con)
        cumulative.append(last_count)

    return KmcTimeline(
        times=times[: len(sequence)],
        cumulative_n2=cumulative,
        energies=energies[: len(sequence)],
        state_ids=sequence,
    )


def plot_kmc_timeline(timeline: KmcTimeline, out_path, log_time: bool = True):
    """Render the two-panel N2-ejection + energy figure to ``out_path``.

    .. versionadded:: 1.8.0
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from chemparseplot.plot.theme import (
            RUHI_COLORS,
            get_theme,
            setup_global_theme,
        )

        setup_global_theme(get_theme("ruhi"))
        teal = RUHI_COLORS["teal"]
        coral = RUHI_COLORS["coral"]
    except Exception:  # pragma: no cover - theme is optional
        teal, coral = "#004D40", "#FF655D"

    out_path = Path(out_path)
    times = np.asarray(timeline.times, dtype=float)
    # log scale cannot show t=0; clamp the initial state to the first positive t.
    plot_times = times.copy()
    if log_time and len(plot_times) > 1:
        positive = plot_times[plot_times > 0]
        floor = positive.min() if positive.size else 1.0
        plot_times[plot_times <= 0] = floor

    fig, (ax_n2, ax_e) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    ax_n2.step(
        plot_times,
        timeline.cumulative_n2,
        where="post",
        color=coral,
        marker="o",
        markersize=4,
        label="N$_2$ ejected",
    )
    ax_n2.set_ylabel("cumulative N$_2$ ejected")
    ax_n2.legend(loc="upper left")
    ax_n2.grid(True, alpha=0.3)

    ax_e.plot(
        plot_times,
        timeline.energies,
        color=teal,
        marker="s",
        markersize=3,
        label="energy",
    )
    ax_e.set_ylabel("energy (eV)")
    ax_e.set_xlabel("KMC time (s)")
    ax_e.grid(True, alpha=0.3)

    ax_state = ax_e.twinx()
    ax_state.plot(
        plot_times,
        timeline.state_ids,
        color="#1E88E5",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="state id",
    )
    ax_state.set_ylabel("visited state id", color="#1E88E5")

    if log_time:
        ax_e.set_xscale("log")

    fig.suptitle("aKMC: N$_2$ ejection over simulation time")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


@click.command()
@click.argument(
    "akmc_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output figure path (default: <akmc_dir>/kmc_n2.png).",
)
@click.option(
    "--core-threshold",
    type=float,
    default=4.0,
    show_default=True,
    help="Min N2-midpoint to Si-core-COM distance (A) to count as ejected.",
)
@click.option(
    "--nn-cutoff",
    type=float,
    default=1.6,
    show_default=True,
    help="Max N-N separation (A) for a bonded N2 pair.",
)
@click.option(
    "--log-time/--linear-time",
    default=True,
    show_default=True,
    help="Use a log scale for the KMC time axis.",
)
def main(
    akmc_dir: Path,
    out_path: Path | None,
    core_threshold: float,
    nn_cutoff: float,
    log_time: bool,
):
    """Plot N2 ejection vs KMC time for an eOn aKMC directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if out_path is None:
        out_path = akmc_dir / "kmc_n2.png"

    timeline = collect_kmc_n2_timeline(akmc_dir, core_threshold, nn_cutoff)
    if not timeline.state_ids:
        click.echo("No aKMC steps found in dynamics.txt.", err=True)
        return

    written = plot_kmc_timeline(timeline, out_path, log_time=log_time)
    final = timeline.cumulative_n2[-1] if timeline.cumulative_n2 else 0
    click.echo(
        f"Visited {len(timeline.state_ids)} state(s); final N2 ejected = {final}"
    )
    click.echo(f"Wrote {written}")


if __name__ == "__main__":
    main()
