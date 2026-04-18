import datetime
from dataclasses import dataclass, field

import numpy as np

@dataclass(frozen=True, slots=True)
class nebiter:
    """
    A typed record representing an iteration of a Nudged Elastic Band (NEB) calculation.

.. versionadded:: 1.0.0

Parameters
----------
iteration : int
    The iteration number of the NEB calculation.
nebpath : nebpath
    The data for the NEB path at this iteration.

See Also
--------
nebpath : Stores the normalized arclength, actual arclength, and energy data for
    the NEB path.
    """

    iteration: int
    nebpath: "nebpath"


@dataclass(frozen=True, slots=True)
class nebpath:
    """
    A typed record representing the NEB path data.

.. versionadded:: 1.0.0

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
The `nebpath` record is used within the `nebiter` record to store
detailed path information for each NEB iteration.
    """

    norm_dist: float
    arc_dist: float
    energy: float


@dataclass
class DimerOpt:
    """Configuration for a dimer-based saddle point search.

    .. versionadded:: 1.0.0
    """

    saddle: str = "dimer"
    rot: str = "lbfgs"
    trans: str = "lbfgs"


@dataclass
class SpinID:
    """Identifier combining molecule ID and spin state.

    .. versionadded:: 1.0.0
    """

    mol_id: int
    spin: str


@dataclass
class MolGeom:
    """Container for molecular geometry with energy and forces.

    .. versionadded:: 1.0.0
    """

    pos: np.array
    energy: float
    forces: np.array


@dataclass
class SaddleMeasure:
    """Aggregated measurements from a saddle point search.

    .. versionadded:: 1.0.0
    """

    pes_calls: int = 0
    iter_steps: int = 0
    tot_time: float = field(default_factory=lambda: datetime.timedelta(0).total_seconds())
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
