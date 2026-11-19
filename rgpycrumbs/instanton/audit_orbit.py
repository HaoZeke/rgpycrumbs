# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "chemparseplot",
#   "click",
#   "matplotlib",
#   "numpy",
#   "readcon>=0.17.1",
# ]
#
# [tool.uv.sources]
# chemparseplot = { git = "https://github.com/HaoZeke/chemparseplot", rev = "03c0deb1889122e2cf451e0d83c56b65bfe7f7a1" }
# ///
"""Reconstruct and score a spectral-instanton oracle batch."""

import json
from pathlib import Path

import click
import numpy as np
from chemparseplot.parse.spectral_instanton import read_oracle_batch
from chemparseplot.units import Q_

from rgpycrumbs.instanton.orbit import proton_transfer_geometry, reconstruct_orbit


@click.command()
@click.argument("batch", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--beta", required=True, type=float, help="Inverse temperature in Hartree^-1."
)
@click.option("--cosines", required=True, type=int, help="Number of cosine coefficients.")
@click.option("--reference-ev", required=True, type=float, help="Energy zero in eV.")
@click.option("--amu-to-electron-mass", type=float, default=1822.8885, show_default=True)
@click.option(
    "--transfer-ids", type=(int, int, int), help="H, donor, acceptor CON atom IDs."
)
@click.option(
    "--plot", type=click.Path(dir_okay=False, path_type=Path), help="Write a plot."
)
def main(*, batch, beta, cosines, reference_ev, amu_to_electron_mass, transfer_ids, plot):
    """Audit a complete ordered half orbit from oracle sidecars."""
    try:
        data = read_oracle_batch(batch)
        mobile = data.mobile
        positions = data.positions[:, mobile].to("bohr").magnitude
        gradients = -data.forces[:, mobile].to("hartree / bohr").magnitude
        masses = np.repeat(data.masses[mobile].magnitude * amu_to_electron_mass, 3)
        energy = data.energy.to("hartree").magnitude
        report = reconstruct_orbit(
            positions.reshape(len(positions), -1),
            energy,
            gradients.reshape(len(gradients), -1),
            masses,
            beta=beta,
            cosines=cosines,
            reference=Q_(reference_ev, "eV").to("hartree").magnitude,
        )
        geometry = {}
        if transfer_ids is not None:
            if not np.allclose(data.angles.to("degree").magnitude, 90):
                msg = "--transfer-ids currently requires an orthorhombic cell"
                raise ValueError(msg)
            indices = tuple(data.atom_ids.index(atom_id) for atom_id in transfer_ids)
            geometry = proton_transfer_geometry(
                data.positions.to("angstrom").magnitude,
                data.cell.to("angstrom").magnitude,
                indices,
            )
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if plot is not None:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from chemparseplot.plot.spectral_instanton import plot_orbit_energy  # noqa: PLC0415

        fig, ax = plt.subplots()
        plot_orbit_energy(
            ax, np.linspace(0, 0.5, len(energy)), data.energy, Q_(reference_ev, "eV")
        )
        fig.savefig(plot, bbox_inches="tight")
        plt.close(fig)
    click.echo(
        json.dumps(
            {
                "beta_hartree_inverse": beta,
                "cosines": cosines,
                "orbit_nodes": len(energy),
                "reference_ev": reference_ev,
                "action_over_hbar": report.action,
                "kinetic_action_over_hbar": report.kinetic_action,
                "max_mass_weighted_residual": report.residual,
                "max_potential_rise_ev": Q_(report.max_potential_rise, "hartree")
                .to("eV")
                .magnitude,
                "reconstruction_error_bohr": report.reconstruction_error,
                **geometry,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
