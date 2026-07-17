# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Seed eOn dimer (min-mode) saddle searches from a converged NEB band.

Reads a converged NEB band (``neb.con``), locates every interior local maximum
of the per-image energy profile, and writes one eOn ``job=saddle_search`` seed
directory per peak. Each seed dir holds the peak geometry (``pos.con``), an
initial unstable mode (``direction.dat``), a finite-difference displaced
geometry (``displacement.con``), and a dimer ``config.ini``.

The initial mode prefers a matching eOn ``peak{MM}_mode.dat`` (paired by RMSD of
``peak{MM}_pos.con`` against the peak frame) when ``peak_files_dir`` is given;
otherwise it uses the normalized NEB tangent ``pos[i+1] - pos[i-1]``.

.. versionadded:: 1.8.0
.. versionchanged:: 1.9.x
    Seed ``config.ini`` is built from eon-schema L1 models (``write_models_ini``), not a hand-maintained dict.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from readcon import ConFrame, read_con, write_con
from scipy.signal import find_peaks

log = logging.getLogger(__name__)

# Mirror the dimer saddle-search blocks from scripts/dimer_from_peak.py.
_RANDOM_SEED = 706253457
_DISPLACE_MAGNITUDE = 0.01


@dataclass(frozen=True)
class PeakSeed:
    """Summary of one seed directory created for a NEB peak.

    .. versionadded:: 1.8.0
    """

    peak_index: int
    image_index: int
    energy: float
    energy_vs_reactant: float
    seed_dir: Path
    mode_source: str  # "eon_peak" or "neb_tangent"


def _image_energies(frames, neb_con: Path) -> np.ndarray:
    """Return per-image energies, falling back to a sibling ``neb.dat``."""
    energies = [getattr(f, "energy", None) for f in frames]
    if all(e is not None for e in energies):
        return np.asarray(energies, dtype=float)

    # Fallback: eOn writes the energy profile to neb.dat (col index 2). Partial
    # runs leave per-iteration neb_NNN.dat files instead; use the latest.
    candidates = [neb_con.with_name("neb.dat")]
    candidates.append(neb_con.with_name(neb_con.stem + ".dat"))
    iter_dats = sorted(neb_con.parent.glob("neb_[0-9]*.dat"))
    if iter_dats:
        candidates.append(iter_dats[-1])

    for dat in candidates:
        if not dat.is_file():
            continue
        rows = [
            line.split()
            for line in dat.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("img")
        ]
        if len(rows) == len(frames):
            log.info("Read image energies from %s", dat.name)
            return np.asarray([float(r[2]) for r in rows], dtype=float)

    missing = [i for i, e in enumerate(energies) if e is None]
    msg = (
        f"neb.con has no per-image energy for images {missing} and no usable "
        f"sibling neb.dat / neb_NNN.dat next to {neb_con}"
    )
    raise ValueError(msg)


def _neb_tangent(frames, idx: int) -> np.ndarray:
    """Normalized central-difference tangent ``pos[i+1] - pos[i-1]``."""
    lo = max(0, idx - 1)
    hi = min(len(frames) - 1, idx + 1)
    fwd = np.asarray(frames[hi].to_ase().get_positions())
    bwd = np.asarray(frames[lo].to_ase().get_positions())
    tangent = fwd - bwd
    norm = np.linalg.norm(tangent)
    if norm > 0:
        tangent = tangent / norm
    return tangent


def _match_eon_mode(
    peak_atoms, peak_files_dir: Path, rmsd_tol: float
) -> np.ndarray | None:
    """Return the eOn mode whose ``peak*_pos.con`` best matches ``peak_atoms``.

    Pairs by direct (non-aligned) RMSD of atom positions; eOn peak geometries
    share atom ordering and frame with the NEB band, so no reordering is needed.
    """
    ref = np.asarray(peak_atoms.get_positions())
    best: tuple[float, Path] | None = None
    for pos_file in sorted(peak_files_dir.glob("peak*_pos.con")):
        match = re.search(r"peak(\d+)_pos\.con$", pos_file.name)
        if match is None:
            continue
        mode_file = pos_file.with_name(f"peak{match.group(1)}_mode.dat")
        if not mode_file.is_file():
            continue
        cand = np.asarray(read_con(str(pos_file))[0].to_ase().get_positions())
        if cand.shape != ref.shape:
            continue
        rmsd = float(np.sqrt(np.mean(np.sum((cand - ref) ** 2, axis=1))))
        if best is None or rmsd < best[0]:
            best = (rmsd, mode_file)

    if best is None or best[0] > rmsd_tol:
        return None

    mode = np.loadtxt(best[1], dtype=float)
    if mode.shape != ref.shape:
        log.warning("eOn mode %s shape %s != %s; ignoring", best[1], mode.shape, ref.shape)
        return None
    norm = np.linalg.norm(mode)
    if norm > 0:
        mode = mode / norm
    return mode


def _dimer_config_models(settings_name: str, socket: str):
    """L1 models + pot extra for a dimer saddle-search seed (eon-schema).

    Aligns improved-dimer defaults with L2 :class:`eon_schema.api.DimerSpec`
    (method=improved). Requires ``eon-schema>=0.2``.
    """
    try:
        from eon_schema.config import (
            DebugConfig,
            DimerConfig,
            LBFGSConfig,
            MainConfig,
            OptimizerConfig,
            PotentialConfig,
            SaddleSearchConfig,
            SocketNWChemPot,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "seed_dimers requires eon-schema>=0.2.\n"
            "  pip install 'rgpycrumbs[eon]'\n"
            "  # or: pip install 'eon-schema>=0.2'"
        ) from exc

    return (
        MainConfig(job="saddle_search", random_seed=_RANDOM_SEED),
        PotentialConfig(potential="socket_nwchem"),
        SocketNWChemPot(
            unix_socket_path=socket,
            unix_socket_mode=True,
            nwchem_settings=settings_name,
            make_template_input=False,
        ),
        SaddleSearchConfig(
            min_mode_method="dimer",
            displace_magnitude=0.0,
            max_energy=50.0,
        ),
        DimerConfig(
            dimer_improved=True,
            opt_method="cg",
            dimer_remove_rotation=False,
            converged_angle=0.5,
        ),
        OptimizerConfig(
            opt_method="lbfgs",
            max_iterations=1000,
            max_move=0.05,
            convergence_metric="norm",
            converged_force=0.05,
        ),
        LBFGSConfig(
            lbfgs_memory=25,
            lbfgs_inverse_curvature=0.01,
            lbfgs_auto_scale=True,
        ),
        DebugConfig(write_movies=True),
    )


def find_neb_peaks(energies: np.ndarray, prominence: float) -> list[int]:
    """Return interior local-maximum image indices (endpoints excluded).

    .. versionadded:: 1.8.0
    """
    energies = np.asarray(energies, dtype=float)
    peaks, _ = find_peaks(energies, prominence=prominence)
    return [int(p) for p in peaks if 0 < p < len(energies) - 1]


def seed_dimers_from_peaks(
    neb_con,
    out_dir,
    settings_name: str,
    socket: str,
    peak_files_dir=None,
    prominence: float = 0.02,
    rmsd_tol: float = 0.1,
) -> tuple[list[Path], list[PeakSeed]]:
    """Write one eOn dimer saddle-search seed dir per interior NEB peak.

    Parameters
    ----------
    neb_con:
        Path to the converged NEB band (``neb.con``). Per-image energies are
        read from :class:`readcon.ConFrame` ``energy``; if absent a sibling
        ``neb.dat`` is used.
    out_dir:
        Parent directory; each peak gets ``out_dir/peak_<NN>/``.
    settings_name:
        Name of the NWChem ``.nwi`` settings fragment referenced by config.ini.
    socket:
        Unix socket path/name for ``[SocketNWChemPot]``.
    peak_files_dir:
        Optional directory holding eOn ``peak{MM}_pos.con`` + ``peak{MM}_mode.dat``
        pairs; the matching mode (by RMSD) seeds ``direction.dat`` when found.
    prominence:
        ``scipy.signal.find_peaks`` prominence threshold on the energy profile.
    rmsd_tol:
        Max RMSD (Angstrom) for an eOn peak geometry to count as a match.

    Returns
    -------
    seed_dirs, summary:
        List of created seed directories and a list of :class:`PeakSeed`.

    .. versionadded:: 1.8.0
    """
    neb_con = Path(neb_con)
    out_dir = Path(out_dir)
    peak_files_dir = Path(peak_files_dir) if peak_files_dir is not None else None

    frames = read_con(str(neb_con))
    if len(frames) < 3:
        msg = f"NEB band needs >= 3 images to have an interior peak, got {len(frames)}"
        raise ValueError(msg)

    energies = _image_energies(frames, neb_con)
    reactant_energy = float(energies[0])
    peak_indices = find_neb_peaks(energies, prominence)
    log.info("Found %d interior peak(s): %s", len(peak_indices), peak_indices)

    models = _dimer_config_models(settings_name, socket)

    seed_dirs: list[Path] = []
    summary: list[PeakSeed] = []
    for nn, idx in enumerate(peak_indices):
        seed_dir = out_dir / f"peak_{nn:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        peak_frame = frames[idx]
        write_con(str(seed_dir / "pos.con"), [peak_frame])
        peak_atoms = peak_frame.to_ase()

        mode = None
        mode_source = "neb_tangent"
        if peak_files_dir is not None and peak_files_dir.is_dir():
            mode = _match_eon_mode(peak_atoms, peak_files_dir, rmsd_tol)
            if mode is not None:
                mode_source = "eon_peak"
        if mode is None:
            mode = _neb_tangent(frames, idx)

        np.savetxt(seed_dir / "direction.dat", mode, fmt="%18.12f")

        disp = peak_atoms.copy()
        disp.set_positions(peak_atoms.get_positions() + _DISPLACE_MAGNITUDE * mode)
        write_con(str(seed_dir / "displacement.con"), [ConFrame.from_ase(disp)])

        _write_dimer_config(seed_dir / "config.ini", models)

        e_img = float(energies[idx])
        summary.append(
            PeakSeed(
                peak_index=nn,
                image_index=idx,
                energy=e_img,
                energy_vs_reactant=e_img - reactant_energy,
                seed_dir=seed_dir,
                mode_source=mode_source,
            )
        )
        seed_dirs.append(seed_dir)
        log.info(
            "peak_%02d <- image %d  E=%.4f (dE=%.4f vs reactant)  mode=%s",
            nn,
            idx,
            e_img,
            e_img - reactant_energy,
            mode_source,
        )

    return seed_dirs, summary


def _write_dimer_config(path: Path, models: tuple) -> None:
    """Write seed config.ini via eon-schema L1 models."""
    from eon_schema.config import write_models_ini

    write_models_ini(path, *models, validate=True)
