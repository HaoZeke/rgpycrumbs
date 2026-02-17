#!/usr/bin/env python3

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "matplotlib",
#     "pandas",
#     "rich",
#     "seaborn",
#     "ase",
#     "mlflow",
# ]
# ///

import io
import logging
import re
import sys
from pathlib import Path

import ase.data
import ase.io
import click
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from rich.console import Console
from rich.logging import RichHandler

# Project imports
from rgpycrumbs.eon._mlflow.log_params import log_config_ini

CONSOLE = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=CONSOLE,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
    ],
)
log = logging.getLogger("rich")

# --- Regex Patterns for eOn Log Parsing ---

# Matches NEB table lines: 1  0.0000e+00     1.3907e+01          10         8.304
NEB_ITER_RE = re.compile(
    r"^\s+(?P<iter>\d+)\s+(?P<step_size>[-\d.e+]+)\s+(?P<force>[-\d.e+]+)\s+(?P<max_img>\d+)\s+(?P<max_en>[-\d.e+]+)"
)

# Matches Dimer search lines: [Dimer] 1  0.0351005  -0.0018  4.63490e-01  -10.2810  2.010  4.777  2
DIMER_STEP_RE = re.compile(
    r"^\[Dimer\]\s+(?P<step>\d+)\s+(?P<step_size>[-\d.e+]+)\s+(?P<delta_e>[-\d.e+]+)\s+"
    r"(?P<force>[-\d.e+]+)\s+(?P<curvature>[-\d.e+]+)\s+(?P<torque>[-\d.e+]+)\s+"
    r"(?P<angle>[-\d.e+| \-]+)\s+(?P<rots>[\d| \-]+)"
)

# Matches Dimer rotation lines: [IDimerRot] ----- --------- ---------- ------------------ -9.9480 5.731 9.06 1
IDIMER_ROT_RE = re.compile(
    r"^\[IDimerRot\]\s+[\-\s]+\s+[\-\s]+\s+[\-\s]+\s+[\-\s]+\s+(?P<curvature>[-\d.e+]+)\s+"
    r"(?P<torque>[-\d.e+]+)\s+(?P<angle>[-\d.e+| \-]+)\s+(?P<rots>[\d| \-]+)"
)

POT_CALLS_RE = re.compile(r"\[XTB\] called potential (?P<count>\d+) times")


def parse_and_log_metrics(log_file: Path):
    """
    Parses the eOn client log and logs metrics using a global step counter.

    This function tracks transitions between NEB and Dimer searches to provide
    a unified convergence timeline.
    """
    global_step = 0
    total_neb_iters = 0

    with log_file.open("r") as f:
        for line in f:
            # 1. Handle NEB Iterations
            if neb_match := NEB_ITER_RE.match(line):
                global_step += 1
                total_neb_iters += 1
                d = neb_match.groupdict()
                force = float(d["force"])
                energy = float(d["max_en"])

                mlflow.log_metric("neb.iteration", int(d["iter"]), step=global_step)
                mlflow.log_metric("neb.force", force, step=global_step)
                mlflow.log_metric("neb.energy", energy, step=global_step)
                mlflow.log_metric("simulation.max_force", force, step=global_step)

            # 2. Handle Dimer Rotation Steps (Inner loops)
            elif rot_match := IDIMER_ROT_RE.match(line):
                d = rot_match.groupdict()
                # We log rotation metrics but do not increment the global search step
                mlflow.log_metric(
                    "dimer.rot.torque", float(d["torque"]), step=global_step
                )
                mlflow.log_metric(
                    "dimer.rot.curvature", float(d["curvature"]), step=global_step
                )

            # 3. Handle Dimer Search Steps (Translations)
            elif dimer_match := DIMER_STEP_RE.match(line):
                global_step += 1
                d = dimer_match.groupdict()
                force = float(d["force"])

                mlflow.log_metric("dimer.step", int(d["step"]), step=global_step)
                mlflow.log_metric("dimer.force", force, step=global_step)
                mlflow.log_metric(
                    "dimer.curvature", float(d["curvature"]), step=global_step
                )
                mlflow.log_metric("simulation.max_force", force, step=global_step)

            # 4. Final summary metrics
            elif pot_match := POT_CALLS_RE.search(line):
                mlflow.log_metric(
                    "total.potential_calls", int(pot_match.group("count"))
                )

    log.info(f"Processed [magenta]{global_step}[/magenta] total optimization steps.")


def plot_structure_evolution(atoms_list, plot_every=5):
    """Generates a horizontal strip showing atomic configuration changes."""
    if not atoms_list:
        return None
    num_structures = len(atoms_list)
    indices = range(0, num_structures, plot_every)

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.set_title("Atomic Configuration Evolution")
    ax.set_xlim(-1, num_structures)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    for i in indices:
        atoms = atoms_list[i]
        buf = io.BytesIO()
        ase.io.write(buf, atoms, format="png", rotation="-90x, -10y, 0z")
        buf.seek(0)
        img = plt.imread(buf)
        imagebox = OffsetImage(img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (i, 0), frameon=False)
        ax.add_artist(ab)
        ax.text(i, -0.6, f"Iter {i}", ha="center", fontsize=8)

    return fig


@click.command()
@click.option(
    "--log-file", "-l", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config.ini"),
)
@click.option(
    "--traj-file",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    help="Optional trajectory file.",
)
@click.option("--experiment", "-e", default="eOn MMF Search")
@click.option(
    "--track-overrides",
    is_flag=True,
    help="Explicitly track user-defined config overrides.",
)
def main(log_file, config_file, traj_file, experiment, track_overrides):
    """Parses eOn logs and logs metrics, plots, and artifacts to MLflow."""
    mlflow.set_experiment(experiment)

    with mlflow.start_run():
        log.info(f"Analyzing eOn log: [cyan]{log_file}[/cyan]")

        # 1. Log Configuration (Hydrated with defaults)
        if config_file.exists():
            log_config_ini(config_file, track_overrides=track_overrides)

        # 2. Parse and log sequential metrics
        parse_and_log_metrics(log_file)

        # 3. Handle trajectory visualization
        if traj_file:
            try:
                atoms = ase.io.read(traj_file, index=":")
                fig_struct = plot_structure_evolution(atoms)
                if fig_struct:
                    mlflow.log_figure(fig_struct, "plots/structure_evolution.png")
                    plt.close(fig_struct)
            except Exception as e:
                log.error(f"Could not process trajectory file: {e}")

        # 4. Log raw artifacts
        mlflow.log_artifact(str(log_file), "raw_logs")
        if traj_file:
            mlflow.log_artifact(str(traj_file), "raw_logs")

        log.info("[bold green]MLflow session finalized.[/bold green]")


if __name__ == "__main__":
    main()
