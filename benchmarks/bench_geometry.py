"""ASV benchmarks for rgpycrumbs.geom (ASE structure analysis)."""

import numpy as np

from rgpycrumbs.geom.analysis import analyze_structure


def _make_water_cluster(n_molecules, spacing=3.0):
    """Build a cluster of water molecules on a grid."""
    from ase import Atoms

    positions = []
    symbols = []
    # Place molecules on a line with given spacing
    for i in range(n_molecules):
        ox = i * spacing
        # O at center, H offset by ~0.96 A at 104.5 degree angle
        positions.extend(
            [
                [ox, 0.0, 0.0],  # O
                [ox + 0.76, 0.59, 0.0],  # H
                [ox + 0.76, -0.59, 0.0],  # H
            ]
        )
        symbols.extend(["O", "H", "H"])
    return Atoms(symbols=symbols, positions=np.array(positions), cell=[50, 50, 50])


class TimeAnalyzeStructure:
    """Benchmark structure analysis on water clusters."""

    params = [3, 10]
    param_names = ["n_molecules"]

    def setup(self, n_molecules):
        self.atoms = _make_water_cluster(n_molecules)

    def time_analyze_structure(self, n_molecules):
        analyze_structure(self.atoms)
