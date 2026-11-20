"""Combine native i-PI partition reports into a single-path molecular rate."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy", "scipy"]
# ///

import importlib
import json
import re
from pathlib import Path

import click
import numpy as np


def fields(lines, heading):
    for index, line in enumerate(lines):
        if heading in line:
            values = re.split(r"[|()\s]+", lines[index + 1].strip())
            values = [complex(v) for v in values if v]
            if any(v.imag != 0 or not np.isfinite(v.real) for v in values):
                raise ValueError("partition factors must be finite and real")
            return [v.real for v in values]
    raise ValueError(f"missing i-PI partition table {heading}")


def molecular_rate(minimum, orbit):
    """Use the native log(Qvib*N) convention with a shared reactant energy zero."""
    constants = importlib.import_module("scipy.constants")
    rlines, ilines = minimum.splitlines(), orbit.splitlines()
    rtemp, itemp = [
        float(re.search(r"Temperature:\s+([\d.eE+-]+) K", text)[1])
        for text in (minimum, orbit)
    ]
    nbeads = int(re.search(r"Instanton rate\. Nbeads (\d+)", orbit)[1])
    nreact = int(re.search(r"Reactants\. Nbeads (\d+)", minimum)[1])
    if nreact != nbeads or not np.isclose(rtemp, itemp, rtol=1e-12) or itemp <= 0:
        raise ValueError(
            "reactant and orbit require matching temperature and full ring size"
        )
    for text in (minimum, orbit):
        if not re.search(r"ASR:\s+poly\b", text):
            raise ValueError("expected a nonlinear freely rotating molecular system")
    if "Overwriting energy shift with the provided values" not in orbit:
        raise ValueError(
            "instanton postprocessing must use the reactant electronic energy"
        )
    qt_r, qr_r, qv_r = fields(rlines, "Qtras(bohr^-3)")
    _, bn_n, qt_i, qr_i, qv_i, action, _, _ = fields(ilines, "log(Qvib*N)")
    if min(bn_n, qt_i, qr_i, qt_r, qr_r) <= 0:
        raise ValueError("the path norm and partition factors must be positive")
    kelvin = constants.Boltzmann * itemp
    temp_au = kelvin / constants.physical_constants["Hartree energy"][0]
    log = (
        np.log(kelvin / constants.h)
        + 0.5 * np.log(2 * np.pi * bn_n * temp_au)
        + np.log(qt_i / qt_r)
        + np.log(qr_i / qr_r)
        + qv_i
        - qv_r
        - action
    )
    return {
        "temperature_K": itemp,
        "beads": nbeads,
        "path_degeneracy": 1,
        "action_over_hbar": action,
        "BN_times_N_atomic_units": bn_n,
        "rotational_ratio": qr_i / qr_r,
        "translational_ratio": qt_i / qt_r,
        "log_Qvib_instanton_native": qv_i,
        "log_Qvib_reactant": qv_r,
        "ln_rate_per_s": float(log),
        "log10_rate_per_s": float(log / np.log(10)),
        "source_precision": "native i-PI partition tables (three decimal places)",
    }


@click.command()
@click.argument("reactant", type=click.Path(exists=True, path_type=Path))
@click.argument("instanton", type=click.Path(exists=True, path_type=Path))
def main(reactant, instanton):
    """Read native i-PI REACTANT and INSTANTON postprocessor reports."""
    click.echo(
        json.dumps(
            molecular_rate(reactant.read_text(), instanton.read_text()), allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
