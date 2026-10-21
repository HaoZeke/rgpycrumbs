#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["click==8.5.0", "matplotlib==3.11.2", "numpy==2.5.4", "chemparseplot", "rgpycrumbs"]
# [tool.uv.sources]
# chemparseplot = { git = "https://github.com/HaoZeke/chemparseplot", rev = "12d39cceb7d9689fa87593e7a6182839c1f03518" }
# ///
"""Compare verified search exports through a reproducible TOML profile."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import click
import tomllib


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), default=None)
def cli(config, output):
    """Render direct savings and per-reaction failure panels."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from chemparseplot.parse.benchmark import load_benchmark_exports  # noqa: PLC0415
    from chemparseplot.plot.benchmark import (  # noqa: PLC0415
        plot_acceptance_curves,
        plot_cost_frontiers,
        plot_cost_savings,
        plot_performance_profiles,
        plot_reaction_costs,
    )

    profile = tomllib.loads(config.read_text())
    settings = profile["comparison"]
    inputs = [(config.parent / p).resolve() for p in settings["exports"]]
    output = output or config.parent / settings["output"]
    output.mkdir(parents=True, exist_ok=True)
    panels = settings["panels"]
    rows = load_benchmark_exports(inputs, {k: v["count"] for k, v in panels.items()})
    methods = settings["methods"]
    titles = {k: v["title"] for k, v in panels.items()}
    plt.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "savefig.dpi": profile.get("shared", {}).get("dpi", 250),
        }
    )
    fig, stats = plot_cost_savings(rows, titles, methods)
    artifacts = []

    def save(fig, name):
        for suffix in ("pdf", "png"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            artifacts.append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        plt.close(fig)

    save(fig, "method_savings")
    profile_fig, profile_stats = plot_performance_profiles(rows, titles, methods)
    save(profile_fig, "performance_profiles")
    frontier_fig, frontier_stats = plot_cost_frontiers(rows, titles, methods)
    save(frontier_fig, "method_cost_frontiers")
    for panel, title in titles.items():
        save(plot_acceptance_curves(rows, panel, title, methods), f"method_acceptance_{panel}")
        save(plot_reaction_costs(rows, panel, title, methods), f"reaction_costs_{panel}")
    manifest = {
        "inputs": [
            {"path": source, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for source, p in zip(settings["exports"], inputs, strict=True)
        ],
        "profile_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "command_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {
            name: {
                "version": importlib.metadata.version(name),
                "source": json.loads(
                    importlib.metadata.distribution(name).read_text("direct_url.json")
                    or "null"
                ),
            }
            for name in ("rgpycrumbs", "chemparseplot", "matplotlib", "numpy")
        },
        "clock": "Search only; independent validation and shared band preparation excluded.",
        "ratio": "Comparator total / GPRidge total on exactly the same accepted reactions.",
        "uncertainty": "One retained measurement per method and reaction; no timing error bars estimated.",
        "panels": stats,
        "performance_profiles": profile_stats,
        "cost_frontiers": frontier_stats,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    cli()
