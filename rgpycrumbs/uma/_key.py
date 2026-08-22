# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""AOTI package cache key: exact composition, charge, spin, task."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import gcd


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
        if z <= 0:
            raise ValueError(f"atomic number must be positive, got {raw!r}")
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
    multiplicities, an odd count only even ones. A mismatch (for
    example a 23-electron radical declared a singlet) selects an
    unphysical PES from a spin-conditioned model.
    """
    m = int(spin)
    if m < 1:
        raise ValueError(f"spin multiplicity must be >= 1, got {spin!r}")
    ne = electron_count(atomic_numbers, charge=charge)
    if (ne + m) % 2 == 0:
        raise ValueError(
            f"spin multiplicity {m} is impossible for {ne} electrons "
            f"(charge {int(charge)}): parity requires "
            f"{'odd' if ne % 2 == 0 else 'even'} multiplicity"
        )


@dataclass(frozen=True)
class UmaMoleKey:
    """Identity of one AOTI-frozen UMA package."""

    task: str
    charge: int
    spin: int
    counts: tuple[tuple[int, int], ...]

    def slug(self) -> str:
        """Filesystem-safe id, e.g. ``omol-q0-s1-z1.1-z6.1-z7.1``."""
        bits = [self.task, f"q{self.charge}", f"s{self.spin}"]
        bits.extend(f"z{z}.{n}" for z, n in self.counts)
        return "-".join(bits)

    def matches_numbers(self, atomic_numbers: Iterable[int]) -> bool:
        return self.counts == exact_counts(atomic_numbers)


def mole_key(
    atomic_numbers: Iterable[int],
    *,
    charge: int = 0,
    spin: int | None = None,
    task: str = "omol",
) -> UmaMoleKey:
    """Build the package key for a system.

    ``spin=None`` derives the minimal multiplicity from electron
    parity; an explicit ``spin`` is validated against it.
    """
    numbers = [int(z) for z in atomic_numbers]
    if spin is None:
        spin = minimal_spin(numbers, charge=charge)
    else:
        validate_spin(numbers, charge=charge, spin=spin)
    return UmaMoleKey(
        task=str(task),
        charge=int(charge),
        spin=int(spin),
        counts=exact_counts(numbers),
    )
