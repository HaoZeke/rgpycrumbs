#!/usr/bin/env python3
"""Normal modes from an eOn Hessian job as a metatensor TensorMap.

.. versionadded:: 1.11.0

eOn's Hessian job writes ``modes.con``: one frame per eigenvalue of the
mass-weighted Hessian, lowest first, with the unit Cartesian mode in the
readcon ``displacements`` section and ``mode_eigenvalue``, ``hbar_omega`` and
``wavenumber`` in the frame metadata. This gathers them into one TensorMap
with two blocks, so the modes travel with their labels into any metatensor
consumer:

- key ``block=0``, the modes: samples ``atom``, components ``xyz``,
  properties ``mode``. Values in Angstrom, each mode of unit norm.
- key ``block=1``, the spectrum: samples ``mode``, properties ``quantity``,
  where quantity 0 is the eigenvalue in eV / (Angstrom**2 amu), 1 is hbar
  omega in eV and 2 is the wavenumber in cm**-1. An imaginary mode reads
  negative in 1 and 2.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "numpy",
#   "readcon>=0.15.0",
#   "metatensor-core>=0.1",
# ]
# ///

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np

log = logging.getLogger(__name__)

#: Spectrum columns, in the order of the ``quantity`` property.
QUANTITIES = ("mode_eigenvalue", "hbar_omega", "wavenumber")


def read_modes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Modes ``(n_atoms, 3, n_modes)`` and spectrum ``(n_modes, 3)``."""
    from readcon import read_con  # noqa: PLC0415

    frames = read_con(str(path))
    if not frames:
        msg = f"{path} holds no frames"
        raise ValueError(msg)
    modes, spectrum = [], []
    for k, frame in enumerate(frames):
        disp = frame.disp
        if disp is None:
            msg = f"frame {k} of {path} has no displacements section"
            raise ValueError(msg)
        md = frame.metadata
        missing = [q for q in QUANTITIES if q not in md]
        if missing:
            msg = f"frame {k} of {path} lacks {', '.join(missing)}"
            raise ValueError(msg)
        modes.append(np.asarray(disp, dtype=np.float64))
        spectrum.append([float(md[q]) for q in QUANTITIES])
    shapes = {m.shape for m in modes}
    if len(shapes) != 1:
        msg = f"frames of {path} differ in atom count: {sorted(shapes)}"
        raise ValueError(msg)
    return np.stack(modes, axis=-1), np.asarray(spectrum)


def modes_tensormap(modes: np.ndarray, spectrum: np.ndarray):
    """The two-block TensorMap described in the module docstring."""
    from metatensor import Labels, TensorBlock, TensorMap  # noqa: PLC0415

    n_atoms, _, n_modes = modes.shape
    mode_labels = Labels(["mode"], np.arange(n_modes, dtype=np.int32)[:, None])
    displacement = TensorBlock(
        values=np.ascontiguousarray(modes),
        samples=Labels(["atom"], np.arange(n_atoms, dtype=np.int32)[:, None]),
        components=[Labels(["xyz"], np.arange(3, dtype=np.int32)[:, None])],
        properties=mode_labels,
    )
    values = TensorBlock(
        values=np.ascontiguousarray(spectrum),
        samples=mode_labels,
        components=[],
        properties=Labels(
            ["quantity"], np.arange(len(QUANTITIES), dtype=np.int32)[:, None]
        ),
    )
    keys = Labels(["block"], np.array([[0], [1]], dtype=np.int32))
    return TensorMap(keys, [displacement, values])


@click.command()
@click.argument("modes_con", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=Path("modes.mts"),
    show_default=True,
    help="metatensor file to write.",
)
def main(modes_con, output):
    """Write the normal modes in MODES_CON as a metatensor TensorMap."""
    import metatensor  # noqa: PLC0415

    logging.basicConfig(level="INFO", format="%(message)s")
    modes, spectrum = read_modes(modes_con)
    metatensor.save(str(output), modes_tensormap(modes, spectrum))
    log.info(
        "%d modes of %d atoms to %s; lowest %.3f cm^-1",
        modes.shape[2],
        modes.shape[0],
        output,
        spectrum[0, 2],
    )


if __name__ == "__main__":
    main()
