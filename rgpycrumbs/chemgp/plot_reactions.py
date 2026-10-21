#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy", "matplotlib>=3.8", "scipy", "ase", "polars", "h5py", "readcon>=0.17.1", "xyzrender>=0.3.8", "chemparseplot", "rgpycrumbs"]
# [tool.uv.sources]
# chemparseplot = { git = "https://github.com/HaoZeke/chemparseplot", branch = "main" }
# readcon = { git = "https://github.com/lode-org/readcon-core", branch = "main" }
# ///
"""Render retained reaction geometries and single-point electronic diagnostics."""

from __future__ import annotations

import csv
import importlib.metadata
import json
from pathlib import Path

import click
import tomllib


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--xtb", default="xtb", show_default=True)
def cli(config, output, xtb):
    """Read a reaction profile; retain calculations, plots and input hashes."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from chemparseplot.parse.eon.con_io import read_first_atoms  # noqa: PLC0415
    from chemparseplot.plot.reaction import (  # noqa: PLC0415
        align_reaction,
        changed_bonds,
        plot_reaction_atlas,
        read_wbo,
        render_reaction_state,
    )

    from rgpycrumbs.geom._xtb import sha256, single_point  # noqa: PLC0415

    profile = tomllib.loads(config.read_text())
    settings = profile["reactions"]
    output = (output or config.parent / settings["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases, records, bonds = [], [], []
    for case in settings["cases"]:
        states = case["states"]
        if len(states) != 4:
            raise ValueError(
                "A reaction row requires reactant, saddle, product and comparator"
            )
        sources = [(config.parent / s["path"]).resolve() for s in states]
        atoms = [read_first_atoms(p) for p in sources]
        pairs = case.get("bonds") or changed_bonds(atoms[0], atoms[2])
        aligned = align_reaction(atoms)
        row = {"title": case["title"], "states": []}
        for i, (spec, frame, source) in enumerate(
            zip(states, aligned, sources, strict=True)
        ):
            target = output / case["id"] / str(i)
            record = single_point(
                frame,
                target,
                charge=case["charge"],
                multiplicity=case["multiplicity"],
                executable=xtb,
                density=settings.get("density", True),
            )
            record = {**record, "source": str(source), "source_sha256": sha256(source)}
            records.append(record)
            state = {
                "title": spec["title"],
                "bonds": [],
                "image": str(target / "structure.png"),
            }
            if "record" in spec:
                validation_path = (config.parent / spec["record"]).resolve()
                document = json.loads(validation_path.read_text())
                validation = document[spec["validation_key"]]
                state["validation"] = validation
                record["validation_record"] = str(validation_path)
                record["validation_record_sha256"] = sha256(validation_path)
                state["subtitle"] = (
                    f"{validation['n_imaginary']} negative; {validation['n_unresolved_vibrational_modes']} unresolved\nFmax = {validation['max_force_eV_per_A']:.3f} eV/Å"
                )
            wbo = read_wbo(target / "wbo", len(frame))
            for a, b in pairs:
                key = tuple(sorted((a, b)))
                bond = {
                    "case": case["id"],
                    "state": spec["title"],
                    "atom_i": a,
                    "atom_j": b,
                    "label": f"{frame[a - 1].symbol}{a}\u2013{frame[b - 1].symbol}{b}",
                    "distance_A": float(frame.get_distance(a - 1, b - 1)),
                    "wbo": wbo.get(key),
                }
                bonds.append(bond)
                state["bonds"].append(bond)
            render_reaction_state(
                target / "geometry.xyz",
                state["image"],
                charge=case["charge"],
                multiplicity=case["multiplicity"],
            )
            if settings.get("density", True):
                state["density_image"] = str(target / "density.png")
                render_reaction_state(
                    target / "geometry.xyz",
                    state["density_image"],
                    cube=target / "density.cub",
                    charge=case["charge"],
                    multiplicity=case["multiplicity"],
                    iso=settings.get("iso", 0.02),
                )
            row["states"].append(state)
        cases.append(row)
    with (output / "bonds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case",
                "state",
                "atom_i",
                "atom_j",
                "label",
                "distance_A",
                "wbo",
            ],
        )
        writer.writeheader()
        writer.writerows(bonds)
    plt.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "savefig.dpi": profile.get("shared", {}).get("dpi", 300),
        }
    )
    for density in [False, True] if settings.get("density", True) else [False]:
        fig = plot_reaction_atlas(cases, density=density)
        name = "reaction_density" if density else "reaction_geometries"
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "profile_sha256": sha256(config),
                "calculations": records,
                "packages": {
                    n: importlib.metadata.version(n)
                    for n in (
                        "rgpycrumbs",
                        "chemparseplot",
                        "xyzrender",
                        "ase",
                        "readcon",
                    )
                },
                "interpretation": "GFN2-xTB single points at rigidly aligned benchmark geometries; no optimization. Density surfaces are qualitative. Saddle acceptance uses the original benchmark potential.",
                "density_isovalue_e_per_bohr3": settings.get("iso", 0.02),
                "artifacts": {
                    p.name: sha256(p)
                    for p in output.iterdir()
                    if p.is_file() and p.name != "manifest.json"
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    cli()
