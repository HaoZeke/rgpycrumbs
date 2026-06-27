"""Shared CLI helpers for single-ended eOn plotting commands."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def default_output_path(prefix: str, plot_type: str, output: Path | None) -> Path:
    """Return the explicit output path or the standard default."""

    if output is not None:
        return output
    return Path(f"{prefix}_{plot_type}.pdf")


def overlay_labels(job_dirs: Sequence[Path], labels: Sequence[str]) -> list[str]:
    """Pad missing overlay labels with job-dir names."""

    resolved = list(labels)
    if len(resolved) < len(job_dirs):
        resolved.extend(path.name for path in job_dirs[len(resolved) :])
    return resolved


def load_trajectories(
    job_dirs: Iterable[Path],
    loader: Callable[[Path], T],
    *,
    log_info: Callable[[str, object, object], None],
    noun: str,
    detail: Callable[[T], str],
) -> list[T]:
    """Load trajectories and emit consistent informational logs."""

    loaded = []
    for path in job_dirs:
        traj = loader(path)
        log_info("Loaded %s from %s (%s)", noun, path, detail(traj))
        loaded.append(traj)
    return loaded
