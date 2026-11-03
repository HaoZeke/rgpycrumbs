#!/usr/bin/env python3
"""GFN2-xTB energies and forces for eOn's ``ext_pot`` potential.

eOn writes ``from_eon_to_extpot`` in the working directory: three box rows,
then ``Z x y z`` for each atom in Angstrom. This program answers in
``from_extpot_to_eon`` with the energy in eV, then one force row per atom in
eV/Angstrom. The molecule is treated as isolated, so the box is not used.
"""

from pathlib import Path

import numpy as np
from ase.units import Bohr, Hartree
from tblite.interface import Calculator


def main() -> None:
    rows = [
        [float(value) for value in line.split()]
        for line in Path("from_eon_to_extpot").read_text().splitlines()
        if line.strip()
    ]
    atoms = np.array(rows[3:])
    numbers = atoms[:, 0].astype(int)
    calc = Calculator("GFN2-xTB", numbers, atoms[:, 1:] / Bohr)
    calc.set("verbosity", 0)
    result = calc.singlepoint()
    energy = result.get("energy") * Hartree
    forces = -result.get("gradient") * Hartree / Bohr
    lines = [f"{energy:.12f}"]
    lines += [f"{fx:.12f} {fy:.12f} {fz:.12f}" for fx, fy, fz in forces]
    Path("from_extpot_to_eon").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
