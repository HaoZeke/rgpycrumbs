"""Compare absolute rate calculations against digitized Arrhenius markers."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["ase", "readcon", "click", "numpy", "scipy", "matplotlib", "chemparseplot @ git+https://github.com/HaoZeke/chemparseplot@ce577ca547fe1ea3904c44e6ef2a277f18789715"]
# ///

import csv
import json
from pathlib import Path

import click
import numpy as np


def draw_thermal_orbit(ax, geometry, ring_file, level_file, label):
    """Overlay the equilibrium molecule and the closed paths of its atoms."""
    from ase.io import read
    from ase.neighborlist import natural_cutoffs, neighbor_list
    from scipy.spatial.transform import Rotation

    atoms = read(geometry, format="extxyz")
    mass = atoms.get_masses()
    reactant = atoms.positions - atoms.get_center_of_mass()
    level = json.loads(level_file.read_text())
    if ring_file.suffix == ".con":
        import readcon

        frames = list(readcon.iter_con(str(ring_file)))
        if not frames:
            raise ValueError("empty readcon orbit")
        half = frames[0].metadata.get("evaluation_counts", {}).get("mode") == "rate"
        key = "neb_bead" if half else "tau_over_beta"
        frames.sort(key=lambda f: f.metadata[key])
        positions = []
        for frame in frames:
            order = np.argsort([atom.atom_id for atom in frame.atoms])
            if ([frame.atoms[i].symbol for i in order] != atoms.get_chemical_symbols()
                    or not np.isclose(frame.metadata["temperature_K"], level["temperature_K"])):
                raise ValueError("orbit identity or temperature mismatch")
            positions.append(np.asarray(frame.xyz)[order])
        ring = np.stack(positions)
        if half:
            ring = np.concatenate((ring, ring[::-1]))
    else:
        ring = np.asarray(json.loads(ring_file.read_text()), dtype=float).reshape(-1, len(atoms), 3)
    if len(ring) != level["beads"]:
        raise ValueError("inset orbit and normalized level specify different bead counts")
    center = np.average(ring.mean(axis=0), axis=0, weights=mass)
    ring -= center
    rotation, _ = Rotation.align_vectors(reactant, ring.mean(axis=0), weights=mass)
    ring = rotation.apply(ring.reshape(-1, 3)).reshape(ring.shape)
    # Show the projected molecular frame from the two directions with largest spatial extent.
    _, _, axes = np.linalg.svd(np.vstack((ring.reshape(-1, 3), reactant)), full_matrices=False)
    reference = reactant @ axes[:2].T
    orbit = ring @ axes[:2].T
    i, j = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.15))
    for a, b in zip(i[i < j], j[i < j]):
        ax.plot(*reference[[a, b]].T, color="#6095bb", lw=1)
    for path in orbit.transpose(1, 0, 2):
        ax.plot(*np.vstack((path, path[0])).T, color="#666666", lw=1.25)
    ax.scatter(*reference.T, s=8 + 2 * atoms.numbers, color="#144a70", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{label} {level['temperature_K']:.1f} K".strip(), fontsize="small")


@click.command()
@click.argument("reference", type=click.Path(exists=True, path_type=Path))
@click.argument("harmonic", type=click.Path(exists=True, path_type=Path))
@click.option("--series", multiple=True, type=(str, click.Path(exists=True, path_type=Path)),
              help="Method name and normalized JSON record or JSONL series. Repeat for each calculation.")
@click.option("--plot", required=True, type=click.Path(path_type=Path))
@click.option("--spectra", type=(click.Path(exists=True, path_type=Path), click.Path(exists=True, path_type=Path)),
              help="Reactant and saddle frequency records for continuous harmonic curves.")
@click.option("--inverse-range", nargs=2, type=float, help="Lower and upper 1000 K/T bounds.")
@click.option("--orbit", type=(click.Path(exists=True, path_type=Path), click.Path(exists=True, path_type=Path),
                               click.Path(exists=True, path_type=Path)),
              help="Equilibrium extxyz, readcon orbit or full bead JSON and normalized level for the inset.")
@click.option("--orbit-label", default="", help="Method used to calculate the displayed orbit.")
def main(reference, harmonic, series, plot, spectra, inverse_range, orbit, orbit_label):
    """Plot absolute log10 rates in inverse seconds and write per-point discrepancies."""
    import matplotlib.pyplot as plt
    from chemparseplot.plot.kinetics import plot_arrhenius
    from chemparseplot.units import Q_

    markers = np.genfromtxt(reference, delimiter=",", names=True, ndmin=1)
    xref = markers["inverse_temperature_1000_per_K"]
    yref = markers["log10_rate_per_s"]
    if not np.all(np.diff(xref) > 0):
        raise ValueError("reference inverse temperatures must increase")
    baseline = np.genfromtxt(harmonic, delimiter=",", names=True, ndmin=1)
    fig, ax = plt.subplots(layout="constrained")
    temperatures = baseline["temperature_K"]
    harmonic_rates = {key: baseline[key] for key in ("htst", "qq_htst")}
    if spectra:
        from rgpycrumbs.kinetics import harmonic_log_rates

        bounds = inverse_range or (min(xref.min(), 1000 / temperatures.max()),
                                   max(xref.max(), 1000 / temperatures.min()))
        if not 0 < bounds[0] < bounds[1]:
            raise ValueError("inverse-temperature bounds must be positive and ordered")
        temperatures = 1000 / np.linspace(*bounds, 250)
        minimum, saddle = (json.loads(path.read_text()) for path in spectra)
        rates = harmonic_log_rates(minimum["signed_wavenumbers_cm-1"], saddle["signed_wavenumbers_cm-1"],
                                   saddle["electronic_energy_eV"] - minimum["electronic_energy_eV"], temperatures)
        harmonic_rates = dict(zip(("htst", "qq_htst"), np.asarray(rates) / np.log(10)))
    if orbit:
        draw_thermal_orbit(ax.inset_axes([0.61, 0.63, 0.35, 0.33]), *orbit, orbit_label)
    for label, key in (("HTST", "htst"), ("qq-HTST", "qq_htst")):
        plot_arrhenius(ax, Q_(temperatures, "kelvin"), harmonic_rates[key],
                       label=label, linestyle="--" if key == "qq_htst" else "-")
    plot_arrhenius(ax, Q_(1000 / xref, "kelvin"), yref, label="Published HQTST",
                   marker="D", color="black", fillstyle="none", linestyle="none")
    table = csv.writer(click.get_text_stream("stdout"))
    table.writerow(("method", "temperature_K", "log10_rate_per_s", "published_interpolated", "difference_decades"))
    by_method = {}
    for label, path in series:
        raw = path.read_text()
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        by_method.setdefault(label, []).extend(rows if isinstance(rows, list) else [rows])
    for label, rows in by_method.items():
        points = sorted((row["temperature_K"], row["log10_rate_per_s"]) for row in rows)
        temperatures, rates = np.asarray(points, dtype=float).T
        if not np.all(np.isfinite(rates) & np.isfinite(temperatures) & (temperatures > 0)):
            raise ValueError(f"invalid absolute rate or temperature in {label}")
        plot_arrhenius(ax, Q_(temperatures, "kelvin"), rates, marker="o", label=label)
        for t, rate in points:
            x = 1000 / t
            match = np.interp(x, xref, yref) if xref[0] <= x <= xref[-1] else None
            table.writerow((label, t, rate, match, rate - match if match is not None else None))
    ax.legend()
    fig.savefig(plot)
    plt.close(fig)


if __name__ == "__main__":
    main()
