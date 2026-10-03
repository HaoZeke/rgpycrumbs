# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Report the export interpreter's compiler side as one JSON line.

Run as ``<python> _probe.py <device>`` by :mod:`rgpycrumbs.uma._env`, in
the interpreter that will run the exporter, so the versions and the
target are the ones the package is compiled with.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

import torch
from torch._inductor import codecache
from torch.utils import cpp_extension

#: Package metadata fields that make up ``ExportEnv.target``; the same
#: tuple as ``rgpycrumbs.uma._key.TARGET_FIELDS``, repeated because this
#: file runs in the export interpreter without importing rgpycrumbs.
TARGET_FIELDS = ("AOTI_MACHINE", "AOTI_CPU_ISA", "AOTI_COMPUTE_CAPABILITY")


def _target(device: str) -> str:
    """The device fields torch writes into the package metadata, joined."""
    kind = device.split(":", 1)[0]
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device cuda requested, torch.cuda.is_available() is False"
            )
        if ":" in device:
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        if cpp_extension.CUDA_HOME is None:
            raise RuntimeError(
                "device cuda: AOTInductor compiles with nvcc, and no CUDA "
                "toolkit was found; set CUDA_HOME or put nvcc on PATH"
            )
    info = codecache.get_device_information(kind)
    return "|".join(str(info.get(f, "")) for f in TARGET_FIELDS)


def main(device: str) -> dict:
    stamps = {}
    for module in (torch,):
        site = Path(module.__file__).resolve().parent.parent
        stamps[str(site)] = site.stat().st_mtime_ns
    try:
        fairchem = metadata.version("fairchem-core")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("fairchem-core is not installed in this interpreter") from exc
    return {
        "device": device.split(":", 1)[0],
        "target": _target(device),
        "torch": metadata.version("torch"),
        "fairchem": fairchem,
        "stamps": stamps,
    }


if __name__ == "__main__":
    try:
        payload = main(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(payload))
