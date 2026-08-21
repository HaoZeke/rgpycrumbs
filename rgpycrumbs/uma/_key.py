# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""merge_mole cache key: reduced stoichiometry, charge, spin, task."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import gcd


def reduced_counts(atomic_numbers: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Return gcd-reduced ``(Z, count)`` pairs, sorted by Z.

    Fairchem ``merge_mole`` compares *reduced* compositions (counts
    divided by their total). Integer counts divided by ``gcd`` are the
    same equivalence: C2H2 and C4H4 share ``((1, 1), (6, 1))``; C2H4 is
    ``((1, 2), (6, 1))``.
    """
    counts: dict[int, int] = {}
    for raw in atomic_numbers:
        z = int(raw)
        if z <= 0:
            raise ValueError(f"atomic number must be positive, got {raw!r}")
        counts[z] = counts.get(z, 0) + 1
    if not counts:
        raise ValueError("atomic_numbers is empty")
    values = list(counts.values())
    g = values[0]
    for n in values[1:]:
        g = gcd(g, n)
    return tuple(sorted((z, n // g) for z, n in counts.items()))


@dataclass(frozen=True)
class UmaMoleKey:
    """Identity of one merge_mole-frozen UMA package."""

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
        return self.counts == reduced_counts(atomic_numbers)


def mole_key(
    atomic_numbers: Iterable[int],
    *,
    charge: int = 0,
    spin: int = 1,
    task: str = "omol",
) -> UmaMoleKey:
    """Build the merge_mole key for a system."""
    return UmaMoleKey(
        task=str(task),
        charge=int(charge),
        spin=int(spin),
        counts=reduced_counts(atomic_numbers),
    )
