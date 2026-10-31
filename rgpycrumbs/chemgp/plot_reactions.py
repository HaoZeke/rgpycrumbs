#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["click==8.5.0", "numpy==2.5.4", "matplotlib==3.11.2", "scipy==1.18.1", "ase==3.29.0", "polars==2.0.0", "h5py==3.16.0", "readcon==0.17.2", "xyzrender==0.3.8", "resvg-py==0.5.0", "chemparseplot>=1.13.0", "rgpycrumbs"]
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
        plot_reaction_gallery,
        read_wbo,
        render_reaction_state,
    )

    from rgpycrumbs.geom._xtb import sha256, single_point  # noqa: PLC0415

    profile = tomllib.loads(config.read_text())
    settings = profile["reactions"]
    output = (output or config.parent / settings["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # An atlas row renders reactant, saddle, product and comparator; a gallery
    # tile renders the saddle alone and reads the endpoints only to choose the
    # bonds the reaction changes.
    layout = settings.get("layout", "atlas")
    if layout not in ("atlas", "gallery"):
        msg = f"Unknown reaction layout {layout!r}; use atlas or gallery"
        raise ValueError(msg)
    expected, rendered = (4, (0, 1, 2, 3)) if layout == "atlas" else (3, (1,))
    cases, records, bonds = [], [], []
    for case in settings["cases"]:
        states = case["states"]
        if len(states) != expected:
            msg = (
                "A reaction row requires reactant, saddle, product and comparator"
                if layout == "atlas"
                else "A gallery tile requires reactant, saddle and product"
            )
            raise ValueError(msg)
        sources = [(config.parent / s["path"]).resolve() for s in states]
        atoms = [read_first_atoms(p) for p in sources]
        pairs = case.get("bonds") or changed_bonds(atoms[0], atoms[2])
        aligned = align_reaction(atoms)
        row = {"title": case["title"], "subtitle": case.get("subtitle", ""), "states": []}
        for i, (spec, frame, source) in enumerate(
            zip(states, aligned, sources, strict=True)
        ):
            if i not in rendered:
                continue
            target = output / case["id"] / str(i)
            record = single_point(
                frame,
                target,
                charge=case["charge"],
                multiplicity=case["multiplicity"],
                executable=xtb,
                density=settings.get("density", True),
                broydamp=spec.get("broydamp", 0.4),
            )
            record = {
                **record,
                "argv": [Path(record["argv"][0]).name, *record["argv"][1:]],
                "source": spec["path"],
                "source_sha256": sha256(source),
            }
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
                record["validation_record"] = spec["record"]
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
            atom_labels = sorted({index for pair in pairs for index in pair})
            labels = [
                f"{index} {frame[index - 1].symbol}{index}" for index in atom_labels
            ]
            render_reaction_state(
                target / "geometry.xyz",
                state["image"],
                charge=case["charge"],
                multiplicity=case["multiplicity"],
                bond_orders=wbo,
                labels=labels,
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
                    bond_orders=wbo,
                    labels=labels,
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
        if layout == "atlas":
            fig = plot_reaction_atlas(cases, density=density)
            name = "reaction_density" if density else "reaction_geometries"
        else:
            tiles = [
                {
                    **row["states"][0],
                    "title": row["title"],
                    "subtitle": row["subtitle"],
                }
                for row in cases
            ]
            fig = plot_reaction_gallery(
                tiles, columns=settings.get("columns", 4), density=density
            )
            name = "reaction_gallery_density" if density else "reaction_gallery"
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "profile_sha256": sha256(config),
                "command_source_sha256": sha256(__file__),
                "property_source_sha256": sha256(single_point.__code__.co_filename),
                "calculations": records,
                "packages": {
                    n: {
                        "version": importlib.metadata.version(n),
                        "source": json.loads(
                            importlib.metadata.distribution(n).read_text(
                                "direct_url.json"
                            )
                            or "null"
                        ),
                    }
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
