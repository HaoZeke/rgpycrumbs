"""Check the native i-PI rate spectrum against the imaginary-time orbit tangent."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "ipi>=3", "numpy"]
# ///

import contextlib
import hashlib
import io
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import click
import numpy as np
from ipi.engine.simulation import Simulation
from ipi.utils.tools import instanton_postproc as native


def phase_mode(checkpoint, frequencies):
    """Rebuild the molecular fluctuation modes with the native rate operators."""
    root = ET.parse(checkpoint)
    for provider in root.iter("ffdirect"):
        for child in list(provider):
            provider.remove(child)
        provider.set("threaded", "False")
        ET.SubElement(provider, "pes").text = "dummy"
    source = io.StringIO(ET.tostring(root.getroot(), encoding="unicode"))
    simulation = Simulation.load_from_xml(
        source, custom_verbosity="quiet", request_banner=False, read_only=True
    )
    system = simulation.syslist[0]
    if system.motion.options["mode"] != "rate":
        raise ValueError("the checkpoint must contain a rate instanton")
    beads = system.motion.beads
    blocks = native.red2comp(
        system.motion.optarrays["hessian"], beads.nbeads, beads.natoms
    )
    path, count, curvature = native.get_double(
        beads.q, beads.nbeads, beads.natoms, blocks
    )
    mass = np.tile(beads.m3, (2, 1))
    temp = system.ensemble.temp
    omega2 = (temp * count * native.kb / native.hbar) ** 2
    springs = native.SpringMapper.spring_hessian(
        beads.natoms, count, beads.m3[0], omega2, mode="full"
    )
    values, vectors, _ = native.clean_hessian(
        curvature + springs, path, beads.natoms, count, beads.m, mass, "poly", mofi=True
    )
    modes = np.sign(values) * np.sqrt(abs(values)) / native.cm2au
    printed = np.loadtxt(frequencies)
    if len(printed) != modes.size + 6 or np.any(printed[:6] != 0):
        raise ValueError("expected a molecular spectrum with six rigid modes")
    # Reconstructing the Hessian with a different LAPACK can perturb nearly zero frequencies.
    np.testing.assert_allclose(modes, printed[6:], rtol=1e-8, atol=1e-5)

    wave = np.fft.fftfreq(count) * (2j * np.pi)
    tangent = np.fft.ifft(wave[:, None] * np.fft.fft(path, axis=0), axis=0).real
    tangent = (tangent * np.sqrt(mass)).ravel()
    if np.linalg.norm(tangent) == 0:
        raise ValueError("a collapsed orbit has no imaginary-time tangent")
    overlaps = abs(vectors.T @ (tangent / np.linalg.norm(tangent)))
    strongest = int(overlaps.argmax())
    return {
        "temperature_K": float(temp / native.K2au),
        "beads": count,
        "additional_surface_requests": 0,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "native_frequencies_sha256": hashlib.sha256(frequencies.read_bytes()).hexdigest(),
        "max_raw_spectrum_difference_cm_inverse": float(abs(modes - printed[6:]).max()),
        "native_deleted_index": 1,
        "native_deleted_frequency_cm_inverse": float(printed[7]),
        "native_deleted_tangent_overlap": float(overlaps[1]),
        "largest_tangent_overlap_index": strongest,
        "largest_tangent_overlap": float(overlaps[strongest]),
        "reaction_mode_frequency_cm_inverse": float(printed[6]),
        "reaction_mode_tangent_overlap": float(overlaps[0]),
        "negative_modes_after_native_deletion": int(
            np.count_nonzero(np.delete(modes, 1) < 0)
        ),
    }


@click.command()
@click.argument("checkpoint", type=click.Path(exists=True, path_type=Path))
@click.argument("frequencies", type=click.Path(exists=True, path_type=Path))
def main(checkpoint, frequencies):
    """Check the omitted rate mode using a native CHECKPOINT and molecular FREQUENCIES."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = phase_mode(checkpoint, frequencies)
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
