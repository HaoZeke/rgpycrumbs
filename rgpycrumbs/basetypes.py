import datetime
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

# namedtuple for storing NEB iteration data
nebiter = namedtuple("nebiter", ["iteration", "nebpath"])
"""
A namedtuple representing an iteration of a Nudged Elastic Band (NEB) calculation.

Parameters
----------
iteration : int
    The iteration number of the NEB calculation.
nebpath : nebpath namedtuple
    The data for the NEB path at this iteration.

See Also
--------
nebpath : Stores the normalized arclength, actual arclength, and energy data for
    the NEB path.
"""

# namedtuple for storing the NEB path data
nebpath = namedtuple("nebpath", ["norm_dist", "arc_dist", "energy"])
"""
A namedtuple representing the NEB path data.

Parameters
----------
norm_dist : float
    Normalized Arclength (0 to 1), representing the progression along the reaction path.
    Calculated as xcoord2 = arcS[img] / arcS[nim-1].
arc_dist : float
    Actual Arclength at each point along the reaction path. Calculated as
    xcoord = arcS[img] + dx(ii).
energy : float
    Interpolated Energy at each point, calculated using cubic polynomial
    interpolation. The energy is calculated using the formula:
    p = a*pow(dx(ii), 3.0) + b*pow(dx(ii), 2.0) + c*dx(ii) + d,
    where a, b, c, and d are coefficients of the cubic polynomial.

Notes
-----
The `nebpath` namedtuple is used within the `nebiter` namedtuple to store
detailed path information for each NEB iteration.
"""


@dataclass
class DimerOpt:
    saddle: str = "dimer"
    rot: str = "lbfgs"
    trans: str = "lbfgs"


@dataclass
class SpinID:
    mol_id: int
    spin: str


@dataclass
class MolGeom:
    pos: np.array
    energy: float
    forces: np.array


@dataclass
class SaddleMeasure:
    pes_calls: int = 0
    iter_steps: int = 0
    tot_time: datetime.timedelta = datetime.timedelta(0).total_seconds()
    saddle_energy: float = np.nan
    saddle_fmax: float = np.nan
    success: bool = False
    method: str = "not run"
    dimer_rot: str = "n/a"
    dimer_trans: str = "n/a"
    init_energy: float = np.nan
    barrier: float = np.nan
    mol_id: int = np.nan
    spin: str = "unknown"
    scf: float = np.nan
    termination_status: str = "not set"
