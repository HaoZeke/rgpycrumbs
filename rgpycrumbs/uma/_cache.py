# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Sidecar lookup for merge_mole-keyed UMA AOTI packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rgpycrumbs.uma._key import UmaMoleKey


def _counts_from_sidecar(data: dict[str, Any]) -> tuple[tuple[int, int], ...] | None:
    raw = data.get("counts")
    if raw is None:
        raw = data.get("reduced_counts")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        out.append((int(item[0]), int(item[1])))
    return tuple(sorted(out))


def key_from_sidecar(data: dict[str, Any]) -> UmaMoleKey | None:
    """Parse a sidecar dict. Returns None if reduced counts are missing."""
    counts = _counts_from_sidecar(data)
    if counts is None:
        return None
    try:
        return UmaMoleKey(
            task=str(data.get("task_name", data.get("task", "omol"))),
            charge=int(data.get("charge", 0)),
            spin=int(data.get("spin", 1)),
            counts=counts,
        )
    except (TypeError, ValueError):
        return None


def sidecar_dict(key: UmaMoleKey, **extra: Any) -> dict[str, Any]:
    payload = dict(extra)
    payload.update(
        {
            "task_name": key.task,
            "charge": key.charge,
            "spin": key.spin,
            "counts": [list(pair) for pair in key.counts],
            "z_set": [z for z, _n in key.counts],
        }
    )
    return payload


def write_sidecar(pt2: Path, key: UmaMoleKey, **extra: Any) -> Path:
    side = Path(pt2).with_suffix(".json")
    existing: dict[str, Any] = {}
    if side.is_file():
        try:
            loaded = json.loads(side.read_text())
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing.update(extra)
    side.write_text(json.dumps(sidecar_dict(key, **existing), indent=2) + "\n")
    return side


def find_package(cache_dir: Path, key: UmaMoleKey) -> Path | None:
    """Return an existing ``.pt2`` whose sidecar matches ``key``."""
    root = Path(cache_dir)
    if not root.is_dir():
        return None
    preferred = root / f"{key.slug()}.pt2"
    candidates = []
    if preferred.is_file():
        candidates.append(preferred)
    candidates.extend(sorted(p for p in root.glob("*.pt2") if p != preferred))
    for pt2 in candidates:
        side = pt2.with_suffix(".json")
        if not side.is_file():
            continue
        try:
            data = json.loads(side.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found = key_from_sidecar(data)
        if found == key:
            return pt2
    return None
