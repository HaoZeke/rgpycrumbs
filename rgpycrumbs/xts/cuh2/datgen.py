from pathlib import Path

import ase.io
from pypotlib.systems import cu_slab_h2 as cuh2slab

from rgpycrumbs._aux import get_gitroot, getstrform


def get_from_gitroot_con(
    fname="cuh2.con",
    hh_range=None,
    h2slab_range=None,
    n_points=None,
):
    hh_range = hh_range or cuh2slab.PltRange(low=-0.05, high=5)
    h2slab_range = h2slab_range or cuh2slab.PltRange(low=-0.05, high=5)
    n_points = n_points or cuh2slab.PlotPoints(x_npt=40, y_npt=40)
    cuh2_dat = get_gitroot()
    src = cuh2_dat / fname
    if Path(fname).suffix.lower() in {".con", ".convel"}:
        from chemparseplot.parse.eon.con_io import read_first_atoms  # noqa: PLC0415

        cuh2_min = read_first_atoms(src)
    else:
        cuh2_min = ase.io.read(getstrform(src))
    cuh2_min.calc = cuh2slab.CuH2PotSlab()

    true_e_dat = cuh2slab.plt_data(
        cuh2_min,
        hh_range=hh_range,
        h2slab_range=h2slab_range,
        n_points=n_points,
    )
    return true_e_dat
