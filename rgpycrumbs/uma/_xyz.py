# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Minimal XYZ I/O so prepare does not need ASE for the cache path."""

from __future__ import annotations

from pathlib import Path

_Z_TO_SYM = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    26: "Fe",
    35: "Br",
    53: "I",
}
_SYM_TO_Z = {sym.upper(): z for z, sym in _Z_TO_SYM.items()}
# Common remaining symbols for arbitrary systems.
_SYM_TO_Z.update(
    {
        "NA": 11,
        "MG": 12,
        "AL": 13,
        "K": 19,
        "CA": 20,
        "TI": 22,
        "CR": 24,
        "MN": 25,
        "CO": 27,
        "NI": 28,
        "CU": 29,
        "ZN": 30,
        "BR": 35,
        "I": 53,
        "PT": 78,
        "AU": 79,
    }
)


def symbol_to_z(sym: str) -> int:
    key = str(sym).strip().upper()
    if key.isdigit():
        return int(key)
    if key in _SYM_TO_Z:
        return _SYM_TO_Z[key]
    raise ValueError(f"unknown element symbol {sym!r}")


def z_to_symbol(z: int) -> str:
    return _Z_TO_SYM.get(int(z), f"X{int(z)}")


def write_xyz(
    path: Path,
    atomic_numbers: list[int],
    positions: list[tuple[float, float, float]],
    comment: str = "",
) -> Path:
    path = Path(path)
    if len(atomic_numbers) != len(positions):
        raise ValueError("atomic_numbers and positions length mismatch")
    lines = [str(len(atomic_numbers)), comment]
    for z, (x, y, zc) in zip(atomic_numbers, positions, strict=True):
        lines.append(f"{z_to_symbol(z)} {x:.17g} {y:.17g} {zc:.17g}")
    path.write_text("\n".join(lines) + "\n")
    return path


def read_xyz(path: Path) -> tuple[list[int], list[tuple[float, float, float]], str]:
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is not a valid XYZ file")
    n = int(lines[0].split()[0])
    comment = lines[1] if len(lines) > 1 else ""
    numbers: list[int] = []
    positions: list[tuple[float, float, float]] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"bad XYZ line in {path}: {line!r}")
        numbers.append(symbol_to_z(parts[0]))
        positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(numbers) != n:
        raise ValueError(f"{path} declared {n} atoms, found {len(numbers)}")
    return numbers, positions, comment
