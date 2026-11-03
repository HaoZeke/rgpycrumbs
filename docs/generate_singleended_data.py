#!/usr/bin/env python3
"""Compute the eOn runs behind the saddle search and minimization tutorials.

The system is ethane. tblite supplies GFN2-xTB energies and forces, and eOn
reaches them through its ``ext_pot`` interface (``docs/xtb_ext_pot.py``).

1. An eOn minimization relaxes the staggered ethane geometry from ASE's G2 set.
2. The minimum is displaced along its softest vibrational mode, the C-C
   torsion, and an eOn dimer search climbs from there to a saddle.
3. The saddle is offset along the dimer mode in both directions, and an eOn
   minimization relaxes each side.

Steps 2 and 3 displace by eOn's default largest optimizer step. eOn offsets
images from a saddle by the same distance in its process searches. The
tutorial data keeps the saddle search and the minimization of the forward
offset in ``docs/orgmode/tutorials/data/ethane/``.

Run as: pixi run -e fragments python docs/generate_singleended_data.py
The eOn client comes from ``--eonclient``, then ``PATH``, then the pixi
``eonmlflow`` environment.
"""

import os

# Serial tblite gives the same numbers on every run.
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.io import read, write
from ase.vibrations import Vibrations
from tblite.ase import TBLite

ROOT = Path(__file__).resolve().parent.parent
TUTORIAL_DATA = ROOT / "docs" / "orgmode" / "tutorials" / "data"
OUT = TUTORIAL_DATA / "ethane"
EXT_POT = ROOT / "docs" / "xtb_ext_pot.py"
# The same box as the Diels-Alder tutorial data; xtb_ext_pot.py ignores it.
BOX_SOURCE = TUTORIAL_DATA / "diels_alder" / "reactant.con"
# eOn 2.16 defaults (eon.schema): [Optimizer] max_move, which also sets the
# default [Process Search] minimization_offset.
MAX_MOVE = 0.2
# A non-linear molecule has three translations and three rotations.
N_RIGID_MODES = 6
# eOn picks a time-based seed when none is set; a fixed one keeps results.dat stable.
RANDOM_SEED = 0
KEEP = {
    "saddle_search": (
        "config.ini",
        "pos.con",
        "displacement.con",
        "direction.dat",
        "climb.con",
        "saddle.con",
        "mode.dat",
        "results.dat",
    ),
    "minimization": (
        "config.ini",
        "pos.con",
        "minimization.con",
        "min.con",
        "results.dat",
    ),
}


def find_eonclient(explicit: str | None) -> Path:
    candidates = [explicit, shutil.which("eonclient")]
    candidates.append(str(ROOT / ".pixi" / "envs" / "eonmlflow" / "bin" / "eonclient"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise SystemExit(
        "eonclient not found: pass --eonclient or run pixi install -e eonmlflow"
    )


def write_con(path: Path, atoms) -> None:
    write(path, atoms, format="eon")
    # The repository hooks reject trailing spaces and a missing final newline.
    lines = path.read_text().splitlines()
    path.write_text("".join(f"{line.rstrip()}\n" for line in lines))


def con_energy(path: Path) -> float:
    """Energy in eV from the JSON metadata line of an eOn ``.con`` file."""
    return json.loads(path.read_text().splitlines()[1])["energy"]


def read_results(path: Path) -> dict[str, str]:
    # eOn writes "value key" lines; only the timing lines put the key first.
    pairs = (line.rsplit(maxsplit=1) for line in path.read_text().splitlines())
    return {pair[1]: pair[0] for pair in pairs if len(pair) == 2}


def run_eon(job_dir: Path, job: str, eonclient: Path) -> dict[str, str]:
    (job_dir / "config.ini").write_text(
        f"[Main]\njob = {job}\nrandom_seed = {RANDOM_SEED}\n\n"
        "[Potential]\npotential = ext_pot\next_pot_path = ./ext_pot\n\n"
        "[Debug]\nwrite_movies = true\n"
    )
    shutil.copyfile(EXT_POT, job_dir / "ext_pot")
    (job_dir / "ext_pot").chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    with (job_dir / "client.log").open("w") as log:
        subprocess.run(  # noqa: S603
            [str(eonclient)],
            cwd=job_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    results = read_results(job_dir / "results.dat")
    if results.get("termination_reason") != "0":
        reason = results.get("termination_reason_text", "no results.dat")
        raise SystemExit(f"eOn {job} in {job_dir.name} stopped: {reason}")
    return results


def softest_mode(atoms, work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    probe = atoms.copy()
    probe.calc = TBLite(method="GFN2-xTB", verbosity=0)
    vib = Vibrations(probe, name=str(work_dir / "vib"))
    vib.run()
    data = vib.get_vibrations()
    mode = data.get_modes()[N_RIGID_MODES]
    return mode / np.linalg.norm(mode), data.get_frequencies()


def describe(atoms, reference) -> str:
    carbons = np.flatnonzero(atoms.numbers == 6)
    hydrogens = np.flatnonzero(atoms.numbers == 1)
    dist = atoms.get_all_distances()
    bonded = carbons[np.argmin(dist[np.ix_(hydrogens, carbons)], axis=1)]
    torsions = [
        abs((atoms.get_dihedral(h0, carbons[0], carbons[1], h1) + 180) % 360 - 180)
        for h0 in hydrogens[bonded == carbons[0]]
        for h1 in hydrogens[bonded == carbons[1]]
    ]
    upper = np.triu_indices(len(atoms), 1)
    shift = np.abs(np.sort(dist[upper]) - np.sort(reference.get_all_distances()[upper]))
    return (
        f"smallest |H-C-C-H| {min(torsions):5.1f} deg, "
        f"C-C {dist[carbons[0], carbons[1]]:.3f} A, "
        f"largest change of a sorted pair distance from the minimum {shift.max():.4f} A"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eonclient", help="eOn client binary")
    eonclient = find_eonclient(parser.parse_args().eonclient)

    start = molecule("C2H6")
    start.cell = read(BOX_SOURCE, format="eon").cell
    start.center()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        jobs = {
            name: work / name
            for name in ("relax", "saddle_search", "minimization", "minimization_reverse")
        }
        for path in jobs.values():
            path.mkdir()

        write_con(jobs["relax"] / "pos.con", start)
        relax = run_eon(jobs["relax"], "minimization", eonclient)
        minimum = read(jobs["relax"] / "min.con", format="eon")

        mode, freqs = softest_mode(minimum, work)
        displaced = minimum.copy()
        displaced.positions += MAX_MOVE * mode
        write_con(jobs["saddle_search"] / "pos.con", minimum)
        write_con(jobs["saddle_search"] / "displacement.con", displaced)
        np.savetxt(jobs["saddle_search"] / "direction.dat", mode, fmt="%.12f")
        search = run_eon(jobs["saddle_search"], "saddle_search", eonclient)

        saddle = read(jobs["saddle_search"] / "saddle.con", format="eon")
        dimer_mode = np.loadtxt(jobs["saddle_search"] / "mode.dat")
        dimer_mode /= np.linalg.norm(dimer_mode)
        sides = {}
        for name, sign in (("minimization", 1.0), ("minimization_reverse", -1.0)):
            offset = saddle.copy()
            offset.positions += sign * MAX_MOVE * dimer_mode
            write_con(jobs[name] / "pos.con", offset)
            sides[name] = run_eon(jobs[name], "minimization", eonclient)

        e_min = con_energy(jobs["relax"] / "min.con")
        print(f"minimum: E = {e_min:.6f} eV, {relax['total_force_calls']} force calls")
        print(f"  {describe(minimum, minimum)}")
        lowest = ", ".join(f"{abs(f):.1f}" for f in freqs[: N_RIGID_MODES + 2])
        print(f"lowest |frequencies| (cm^-1): {lowest}")
        print(f"dimer start: {describe(displaced, minimum)}")
        e_sp = con_energy(jobs["saddle_search"] / "saddle.con")
        print(
            f"saddle: E = {e_sp:.6f} eV, barrier {e_sp - e_min:.4f} eV, "
            f"eigenvalue {float(search['final_eigenvalue']):.4f} eV/A^2, "
            f"{search['iterations']} iterations, "
            f"{search['total_force_calls']} force calls"
        )
        print(f"  {describe(saddle, minimum)}")
        for name, res in sides.items():
            end = read(jobs[name] / "min.con", format="eon")
            e_end = con_energy(jobs[name] / "min.con")
            print(
                f"{name}: E = {e_end:.6f} eV, {e_end - e_min:+.6f} eV from the minimum, "
                f"{res['total_force_calls']} force calls"
            )
            print(f"  {describe(end, minimum)}")

        for job, names in KEEP.items():
            dest = OUT / job
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            for name in names:
                shutil.copyfile(jobs[job] / name, dest / name)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
