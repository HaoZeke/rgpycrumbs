"""Calculate vibrational rates and compare them with a reference temperature series."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click", "numpy", "scipy", "matplotlib",
#   "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@ce577ca547fe1ea3904c44e6ef2a277f18789715",
# ]
# ///

import json
from pathlib import Path

import click
import numpy as np

from rgpycrumbs.kinetics import harmonic_log_rates


@click.command()
@click.argument("reactant", type=click.Path(exists=True, path_type=Path))
@click.argument("saddle", type=click.Path(exists=True, path_type=Path))
@click.argument("reference", type=click.Path(exists=True, path_type=Path))
@click.option("--plot", type=click.Path(path_type=Path), help="Write the rate comparison figure.")
def main(reactant, saddle, reference, plot):
    """Compare two vibration records with a CSV of reference log10 rates.

    Records contain electronic_energy_eV and signed_wavenumbers_cm-1. The CSV
    columns are temperature_K, htst and qq_htst (log10 rates in inverse seconds).
    """
    minimum, transition = (json.loads(p.read_text()) for p in (reactant, saddle))
    points = np.genfromtxt(reference, delimiter=",", names=True, ndmin=1)
    temperatures = points["temperature_K"]
    rates = harmonic_log_rates(minimum["signed_wavenumbers_cm-1"], transition["signed_wavenumbers_cm-1"],
                               transition["electronic_energy_eV"] - minimum["electronic_energy_eV"], temperatures)
    classical, quantum = np.asarray(rates) / np.log(10)
    table = np.column_stack((temperatures, classical, classical - points["htst"], quantum, quantum - points["qq_htst"]))
    np.savetxt(click.get_text_stream("stdout"), table, delimiter=",", fmt="%.9g", comments="",
               header="temperature_K,htst,htst_difference,qq_htst,qq_htst_difference")
    if plot:
        import matplotlib.pyplot as plt
        from chemparseplot.plot.kinetics import plot_arrhenius
        from chemparseplot.units import Q_

        fig, ax = plt.subplots(layout="constrained")
        for values, key, label in ((classical, "htst", "HTST"), (quantum, "qq_htst", "qq-HTST")):
            curve = plot_arrhenius(ax, Q_(temperatures, "kelvin"), values, label=label)
            plot_arrhenius(ax, Q_(temperatures, "kelvin"), points[key], linestyle="none",
                           marker="o", fillstyle="none", color=curve.get_color(), label=f"{label} reference")
        ax.legend()
        fig.savefig(plot)
        plt.close(fig)


if __name__ == "__main__":
    main()
