# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Probe the export interpreter once, then remember the answer.

The key needs the torch and fairchem versions and the device target of
the interpreter that runs the exporter. Asking costs a torch import
(seconds), which would dominate a cache hit, so the answer is stored
under ``<cache>/.env/``. For the running interpreter the memo is keyed
by its installed distributions; for another interpreter by its path,
and reused while its site-packages directory is unchanged (installing
or removing a distribution changes its mtime).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from rgpycrumbs.uma._key import ExportEnv

_PROBE = Path(__file__).with_name("_probe.py")


def normalize_device(device: str) -> str:
    """``cpu``, ``cuda`` or ``cuda:N``; anything else is refused."""
    text = str(device).strip().lower()
    kind, _, index = text.partition(":")
    if kind == "cpu" and not index:
        return "cpu"
    if kind == "cuda" and (not index or index.isdigit()):
        return text
    raise ValueError(f"device must be cpu, cuda or cuda:N, got {device!r}")


def _local_fingerprint() -> str:
    """Installed distributions of this interpreter, from directory names alone.

    ``uv run --script`` can give one set of packages a new environment
    path on every run, so for the running interpreter the identity is
    what is installed (``<name>-<version>.dist-info`` entries on
    ``sys.path``), not where.
    """
    names = []
    for entry in sys.path:
        try:
            names.extend(n for n in os.listdir(entry or ".") if n.endswith(".dist-info"))
        except OSError:
            continue
    return "\0".join([sys.version, os.path.realpath(sys.executable), *sorted(names)])


def _is_local(python: str) -> bool:
    return os.path.abspath(python) == os.path.abspath(sys.executable)


def _memo_path(cache_dir: Path, python: str, device: str) -> Path:
    where = _local_fingerprint() if _is_local(python) else os.path.abspath(python)
    ident = "\0".join(
        [
            where,
            socket.gethostname(),
            device,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            os.environ.get("CUDA_HOME", ""),
        ]
    )
    digest = hashlib.sha256(ident.encode()).hexdigest()[:20]
    return Path(cache_dir) / ".env" / f"{digest}.json"


def _fresh(memo: dict) -> bool:
    stamps = memo.get("stamps")
    if not isinstance(stamps, dict) or not stamps:
        return False
    for path, mtime in stamps.items():
        try:
            if Path(path).stat().st_mtime_ns != mtime:
                return False
        except OSError:
            return False
    return True


def probe_env(
    cache_dir: Path, *, python: str | None = None, device: str = "cpu"
) -> ExportEnv:
    """Return the :class:`ExportEnv` of ``python`` (default: this interpreter)."""
    python = python or sys.executable
    device = normalize_device(device)
    memo_path = _memo_path(cache_dir, python, device)
    try:
        memo = json.loads(memo_path.read_text())
    except (OSError, ValueError):
        memo = None
    if not (isinstance(memo, dict) and (_is_local(python) or _fresh(memo))):
        proc = subprocess.run(  # noqa: S603
            [python, str(_PROBE), device],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = proc.stdout.strip().splitlines()
        try:
            memo = json.loads(lines[-1]) if lines else {}
        except ValueError:
            memo = {}
        if proc.returncode != 0 or "error" in memo or not memo:
            detail = memo.get("error") or proc.stderr.strip()[-2000:] or "no output"
            raise RuntimeError(
                f"probing {python} for torch and fairchem failed: {detail}"
            )
        memo_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = memo_path.with_name(f".{memo_path.name}.{os.getpid()}")
        tmp.write_text(json.dumps(memo) + "\n")
        os.replace(tmp, memo_path)
    return ExportEnv(
        device=str(memo["device"]),
        target=str(memo["target"]),
        torch=str(memo["torch"]),
        fairchem=str(memo["fairchem"]),
    )
