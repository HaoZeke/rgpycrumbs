# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""Cache key of one UMA AOTI package.

A package is reusable only when every input that shaped the compiled
graph matches: the exact composition, charge, spin multiplicity and task
head (``merge_mole`` folds them into the weights), the checkpoint, the
tracing dtype and the static-shape options (molecular box, band size),
and the compiler side (device, its CPU ISA or CUDA compute capability,
and the torch and fairchem versions).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from math import gcd
from typing import Any

#: Version of the key layout. Bumping it invalidates every cached package.
KEY_SCHEMA = 1

#: Package metadata fields that make up :attr:`ExportEnv.target`.
TARGET_FIELDS = ("AOTI_MACHINE", "AOTI_CPU_ISA", "AOTI_COMPUTE_CAPABILITY")

_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co"
    " Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te"  # codespell:ignore te
    " I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir"  # codespell:ignore nd
    " Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No"
    " Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()


def exact_counts(atomic_numbers: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Return ``(Z, count)`` pairs, sorted by Z, without reduction.

    An AOTI export freezes per-atom buffers at the example system's
    size, so a package compiled for C2H2 aborts inside the compiled
    graph when fed C4H4 even though fairchem ``merge_mole`` treats the
    two as the same reduced composition. The cache key therefore uses
    the exact composition, never the gcd-reduced one.
    """
    counts: dict[int, int] = {}
    for raw in atomic_numbers:
        z = int(raw)
        if not 0 < z < len(_SYMBOLS):
            raise ValueError(f"atomic number out of range: {raw!r}")
        counts[z] = counts.get(z, 0) + 1
    if not counts:
        raise ValueError("atomic_numbers is empty")
    return tuple(sorted(counts.items()))


def reduced_counts(atomic_numbers: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Return gcd-reduced ``(Z, count)`` pairs, sorted by Z.

    Fairchem ``merge_mole`` compares *reduced* compositions: C2H2 and
    C4H4 both reduce to ``((1, 1), (6, 1))``. This is the model-side
    equivalence for expert merging; it is NOT sufficient to reuse an
    AOTI package (see :func:`exact_counts`).
    """
    counts = exact_counts(atomic_numbers)
    g = 0
    for _z, n in counts:
        g = gcd(g, n)
    return tuple((z, n // g) for z, n in counts)


def hill_formula(counts: Iterable[tuple[int, int]]) -> str:
    """Hill-order formula of ``(Z, count)`` pairs, e.g. ``CHN`` or ``C2H2``."""
    by_symbol = {_SYMBOLS[z]: n for z, n in counts}
    order = sorted(by_symbol)
    if "C" in by_symbol:
        head = ["C", "H"] if "H" in by_symbol else ["C"]
        order = head + [s for s in order if s not in head]
    return "".join(s if by_symbol[s] == 1 else f"{s}{by_symbol[s]}" for s in order)


def electron_count(atomic_numbers: Iterable[int], *, charge: int = 0) -> int:
    """Total electrons of the system: sum(Z) - charge."""
    return sum(int(z) for z in atomic_numbers) - int(charge)


def minimal_spin(atomic_numbers: Iterable[int], *, charge: int = 0) -> int:
    """Minimal spin multiplicity consistent with electron parity.

    Even electron count gives a singlet (1); odd gives a doublet (2).
    """
    return 1 if electron_count(atomic_numbers, charge=charge) % 2 == 0 else 2


def validate_spin(
    atomic_numbers: Iterable[int], *, charge: int = 0, spin: int = 1
) -> None:
    """Raise when ``spin`` cannot hold the system's electron count.

    Multiplicity is ``2S + 1``: an even electron count admits only odd
    multiplicities, an odd count only even ones, and ``N`` electrons
    reach at most ``N + 1``. A mismatch (for example a 23-electron
    radical declared a singlet) selects an unphysical PES from a
    spin-conditioned model.
    """
    m = int(spin)
    if m < 1:
        raise ValueError(f"spin multiplicity must be >= 1, got {spin!r}")
    ne = electron_count(atomic_numbers, charge=charge)
    if ne < 0:
        raise ValueError(f"charge {int(charge)} leaves {ne} electrons")
    if (ne + m) % 2 == 0:
        raise ValueError(
            f"spin multiplicity {m} is impossible for {ne} electrons "
            f"(charge {int(charge)}): parity requires "
            f"{'odd' if ne % 2 == 0 else 'even'} multiplicity"
        )
    if m > ne + 1:
        raise ValueError(f"spin multiplicity {m} needs more than {ne} electrons")


def canonical_float(value: float) -> str:
    """Stable text for a float option: ``25.0`` and ``25`` both give ``25``."""
    return format(float(value), ".12g")


@dataclass(frozen=True)
class ExportEnv:
    """Compiler side of a package: what the exporting interpreter builds for.

    ``device`` is ``cpu`` or ``cuda``. ``target`` joins, with ``|``, the
    fields torch records in the package metadata for the machine it
    compiled on (:data:`TARGET_FIELDS`): the architecture, the CPU
    vector ISA and, for CUDA, the compute capability. A package built
    for another target fails to load or runs illegal instructions.
    """

    device: str
    target: str
    torch: str
    fairchem: str


@dataclass(frozen=True)
class AotiKey:
    """Identity of one compiled UMA package."""

    counts: tuple[tuple[int, int], ...]
    charge: int
    spin: int
    task: str
    model: str
    dtype: str
    molecular_box: str
    batch_max: int
    device: str
    target: str
    torch: str
    fairchem: str
    schema: int = KEY_SCHEMA

    @property
    def formula(self) -> str:
        return hill_formula(self.counts)

    @property
    def natoms(self) -> int:
        return sum(n for _z, n in self.counts)

    @property
    def z_set(self) -> list[int]:
        return [z for z, _n in self.counts]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form; :meth:`from_dict` inverts it exactly."""
        data = asdict(self)
        data["counts"] = [list(pair) for pair in self.counts]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AotiKey:
        fields = dict(data)
        fields["counts"] = tuple((int(z), int(n)) for z, n in fields.get("counts", ()))
        return cls(**fields)

    def canonical(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """80-bit hex digest of every field."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:20]

    def stem(self) -> str:
        """File stem: readable prefix plus the digest of the full key.

        ``omol-CHN-q0-s1-uma-s-1p1-cpu-<digest>``. The prefix is for
        people; the digest is what separates two keys.
        """
        bits = [
            self.task,
            self.formula,
            f"q{self.charge}",
            f"s{self.spin}",
            self.model,
            self.device,
            self.digest(),
        ]
        return "-".join(_safe(b) for b in bits)


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._+-" else "_" for c in str(text))


def aoti_key(
    atomic_numbers: Iterable[int],
    env: ExportEnv,
    *,
    charge: int = 0,
    spin: int | None = None,
    task: str = "omol",
    model: str = "uma-s-1p1",
    dtype: str = "float32",
    molecular_box: float = 0.0,
    batch_max: int = 0,
) -> AotiKey:
    """Build the package key for a system and an export environment.

    ``spin=None`` derives the minimal multiplicity from electron
    parity; an explicit ``spin`` is validated against it. ``batch_max``
    of 0 and 1 both mean a single-system package.
    """
    numbers = [int(z) for z in atomic_numbers]
    if spin is None:
        spin = minimal_spin(numbers, charge=charge)
    else:
        validate_spin(numbers, charge=charge, spin=spin)
    if not float(molecular_box) >= 0.0:
        raise ValueError(f"molecular_box must be >= 0, got {molecular_box!r}")
    if int(batch_max) < 0:
        raise ValueError(f"batch_max must be >= 0, got {batch_max!r}")
    return AotiKey(
        counts=exact_counts(numbers),
        charge=int(charge),
        spin=int(spin),
        task=str(task),
        model=str(model),
        dtype=str(dtype),
        molecular_box=canonical_float(molecular_box),
        batch_max=int(batch_max) if int(batch_max) > 1 else 0,
        device=env.device,
        target=env.target,
        torch=env.torch,
        fairchem=env.fairchem,
    )
