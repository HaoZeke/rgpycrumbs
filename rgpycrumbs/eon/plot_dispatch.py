# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Object-aware eOn plot entry: nanobind types and ConFrame sequences.

Kind comes from the object type or an explicit ``kind=`` when only frames are
passed. Never sniffs mixed job directories for ``neb.con`` vs ``min.con``.

```{versionadded} 1.10.4
```
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rgpycrumbs.eon.plot_config import run_plot

__all__ = [
    "adapt_plot_source",
    "plot",
]


def _type_name(obj: Any) -> str:
    return type(obj).__name__


def _module_name(obj: Any) -> str:
    return getattr(type(obj), "__module__", "") or ""


def _is_pyeon_type(obj: Any, *names: str) -> bool:
    """True when obj is a pyeonclient nanobind type (by module + class name)."""
    mod = _module_name(obj)
    if "pyeonclient" not in mod:
        return False
    return _type_name(obj) in names


def _looks_like_conframe(obj: Any) -> bool:
    return (
        hasattr(obj, "energy")
        and hasattr(obj, "to_ase")
        and (hasattr(obj, "metadata") or hasattr(obj, "frame_index"))
    )


def _is_frame_sequence(obj: Any) -> bool:
    if isinstance(obj, (str, bytes, Path)):
        return False
    if not isinstance(obj, Sequence):
        return False
    if len(obj) == 0:
        return False
    return _looks_like_conframe(obj[0])


def _extract_frames_from_object(obj: Any) -> list[Any] | None:
    """Return ConFrame list from known pyeonclient retention APIs, else None."""
    for meth in ("path_frames", "to_conframes", "movie_frames", "climb_frames"):
        fn = getattr(obj, meth, None)
        if callable(fn):
            try:
                frames = fn()
            except Exception:
                # path_frames before compute raises — try next method
                continue
            if frames is not None:
                return list(frames)
    return None


def adapt_plot_source(
    obj: Any,
    *,
    kind: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Map *obj* to ``(command, payload)`` for :func:`run_plot`.

    Parameters
    ----------
    obj
        pyeonclient ``NEB`` / ``NudgedElasticBand`` (after compute),
        ``Matter`` with retained movie frames, ``MinModeSaddleSearch`` with
        climb frames, a sequence of ConFrames, or a pre-built trajectory DTO.
    kind
        Required when *obj* is only a ConFrame sequence: ``neb`` | ``min`` |
        ``saddle``. Ignored when type dispatch is unambiguous.

    Returns
    -------
    command, payload
        *command* is ``neb`` | ``min`` | ``saddle``. *payload* is merged into
        settings (``frames``, ``trajectory``, …).
    """
    # Pre-built chemparseplot DTOs (duck by attributes)
    if (
        hasattr(obj, "atoms_list")
        and hasattr(obj, "dat_df")
        and hasattr(obj, "initial_atoms")
        and hasattr(obj, "final_atoms")
        and not hasattr(obj, "saddle_atoms")
    ):
        return "min", {"trajectory": obj}
    if (
        hasattr(obj, "atoms_list")
        and hasattr(obj, "dat_df")
        and hasattr(obj, "saddle_atoms")
    ):
        return "saddle", {"trajectory": obj}

    # pyeonclient NEB / band
    if _is_pyeon_type(obj, "NEB", "NudgedElasticBand") or (
        callable(getattr(obj, "path_frames", None))
        and callable(getattr(obj, "compute", None))
        and "pyeonclient" in _module_name(obj)
    ):
        frames = _extract_frames_from_object(obj)
        if frames is None:
            # NEB chemist: try .band
            band = getattr(obj, "band", None)
            if band is not None:
                frames = _extract_frames_from_object(band)
        if not frames:
            msg = (
                f"{_type_name(obj)!r} has no path_frames after compute(); "
                "call compute() first"
            )
            raise ValueError(msg)
        return "neb", {"frames": frames}

    # Matter with retained min movie
    if _is_pyeon_type(obj, "Matter") or (
        callable(getattr(obj, "movie_frames", None))
        and callable(getattr(obj, "relax", None))
        and "pyeonclient" in _module_name(obj)
    ):
        frames = list(obj.movie_frames())
        if not frames:
            msg = "Matter.movie_frames() is empty; relax(..., retain_frames=True) first"
            raise ValueError(msg)
        return "min", {"frames": frames}

    # MinModeSaddleSearch climb
    if _is_pyeon_type(obj, "MinModeSaddleSearch") or (
        callable(getattr(obj, "climb_frames", None))
        and callable(getattr(obj, "run_retain_frames", None))
    ):
        frames = list(obj.climb_frames())
        if not frames:
            msg = "climb_frames() is empty; run_retain_frames(...) first"
            raise ValueError(msg)
        return "saddle", {"frames": frames}

    # Bare ConFrame sequence — kind required
    if _is_frame_sequence(obj):
        if kind is None:
            msg = (
                "ConFrame sequence requires kind='neb'|'min'|'saddle' "
                "(no job-dir sniffing)"
            )
            raise TypeError(msg)
        k = kind.strip().lower()
        if k not in {"neb", "min", "saddle"}:
            msg = f"kind must be neb|min|saddle, got {kind!r}"
            raise ValueError(msg)
        return k, {"frames": list(obj)}

    msg = (
        f"plot() does not support {type(obj)!r}. "
        "Pass a pyeonclient NEB/NudgedElasticBand (after compute), "
        "Matter with movie_frames, MinModeSaddleSearch with climb_frames, "
        "a ConFrame sequence with kind=, or a Min/DimerTrajectoryData."
    )
    raise TypeError(msg)


def _runner_for(command: str) -> Callable[[dict[str, Any]], Any]:
    if command == "neb":
        from rgpycrumbs.eon.plt_neb import plot_neb_from_settings

        return plot_neb_from_settings
    if command == "min":
        from rgpycrumbs.eon.plt_min import plot_min_from_settings

        return plot_min_from_settings
    if command == "saddle":
        from rgpycrumbs.eon.plt_saddle import plot_saddle_from_settings

        return plot_saddle_from_settings
    msg = f"unknown plot command {command!r}"
    raise ValueError(msg)


def plot(
    obj: Any,
    *,
    kind: str | None = None,
    config: str | Path | None = None,
    **overrides: Any,
) -> Path | None:
    """Library plot entry for live eOn objects or ConFrame sequences.

    Examples
    --------
    >>> plot(neb, plot_type="profile", output_file="1d.pdf")  # after compute
    >>> plot(frames, kind="neb", plot_type="profile", output_file="1d.pdf")
    >>> plot(matter, plot_type="profile", output="min.pdf")  # after retain_frames
    """
    from rgpycrumbs._aux import enable_library_auto_deps

    enable_library_auto_deps()
    command, payload = adapt_plot_source(obj, kind=kind)
    # frames → trajectory DTOs for min/saddle (chemparseplot)
    if "frames" in payload and command in {"min", "saddle"}:
        frames = payload.pop("frames")
        if command == "min":
            from chemparseplot.parse.eon.frame_series import (
                min_trajectory_from_frames,
            )

            payload["trajectory"] = min_trajectory_from_frames(frames)
        else:
            from chemparseplot.parse.eon.frame_series import (
                dimer_trajectory_from_frames,
            )

            payload["trajectory"] = dimer_trajectory_from_frames(frames)

    merged = {**payload, **overrides}
    return run_plot(command, _runner_for(command), config=config, **merged)
