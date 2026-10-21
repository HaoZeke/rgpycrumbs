"""Retained single-point xTB diagnostics for molecular figures."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def single_point(
    atoms, directory, *, charge, multiplicity, executable="xtb", timeout=180, density=True
):
    """Evaluate the supplied geometry without optimizing or changing its spin."""
    import io  # noqa: PLC0415

    from ase.io import write  # noqa: PLC0415

    if multiplicity < 1 or int(multiplicity) != multiplicity or int(charge) != charge:
        raise ValueError(
            "Charge and multiplicity must be integers; multiplicity is positive"
        )
    if atoms.pbc.any():
        raise ValueError("This molecular xTB diagnostic requires nonperiodic coordinates")
    nelectrons = int(atoms.numbers.sum()) - charge
    unpaired = multiplicity - 1
    if unpaired > nelectrons or (nelectrons - unpaired) % 2:
        raise ValueError("Charge and multiplicity disagree with the electron count")
    stream = io.StringIO()
    write(stream, atoms, format="xyz")
    geometry = stream.getvalue()
    settings = {
        "geometry_sha256": hashlib.sha256(geometry.encode()).hexdigest(),
        "charge": charge,
        "multiplicity": multiplicity,
        "gfn": 2,
        "density": density,
        "cube_step_bohr": 0.3,
        "cube_density_matrix_threshold": 0.0,
    }
    directory = Path(directory)
    record = directory / "properties.json"
    if record.exists():
        retained = json.loads(record.read_text())
        if retained["settings"] != settings:
            raise ValueError(f"Stored property calculation differs: {directory}")
        for name, digest in retained["files"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Stored property artifact changed: {directory / name}")
        return retained
    directory.mkdir(parents=True, exist_ok=False)
    binary = shutil.which(executable)
    if binary is None:
        raise FileNotFoundError(f"xTB executable not found: {executable}")
    (directory / "geometry.xyz").write_text(geometry)
    control = "$write\n  wiberg=true\n"
    if density:
        control += "  density=true\n$cube\n  step=0.3\n  pthr=0.0\n"
    (directory / "xtb.inp").write_text(control + "$end\n")
    command = [
        binary,
        "geometry.xyz",
        "--gfn",
        "2",
        "--chrg",
        str(charge),
        "--uhf",
        str(unpaired),
        "--sp",
        "--wbo",
        "--input",
        "xtb.inp",
    ]
    version = subprocess.run(  # noqa: S603
        [binary, "--version"], capture_output=True, text=True, check=True, timeout=20
    )
    with (directory / "xtb.log").open("w") as log:
        process = subprocess.run(  # noqa: S603
            command,
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"xTB exited {process.returncode}; see {directory / 'xtb.log'}"
        )
    names = ["geometry.xyz", "xtb.inp", "xtb.log", "wbo"] + (
        ["density.cub"] if density else []
    )
    result = {
        "settings": settings,
        "argv": command,
        "returncode": process.returncode,
        "version": version.stdout + version.stderr,
        "executable_sha256": sha256(binary),
        "files": {name: sha256(directory / name) for name in names},
    }
    record.write_text(json.dumps(result, indent=2) + "\n")
    return result
