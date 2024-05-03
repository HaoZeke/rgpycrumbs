import ast
import re
import subprocess
from pathlib import Path

import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
from pypotlib.systems import cu_slab_h2 as cuh2slab
from pyprotochemgp.systems.cuh2slab import prepare_scatter_points


cuh2_dat = gitroot
cuh2_min = ase.io.read(getstrform(cuh2_dat / "cuh2.con"))
cuh2_min.calc = cuh2slab.CuH2PotSlab()

true_e_dat = cuh2slab.plt_data(
    cuh2_min,
    hh_range=cuh2slab.PltRange(low=-0.05, high=5),
    h2slab_range=cuh2slab.PltRange(low=-0.05, high=5),
    n_points=cuh2slab.PlotPoints(x_npt=40, y_npt=40),
)


def plot_band(_index, _band, _k, _method="xts", _opt="SD", _ci="False"):
    plot_last = _band
    cuh2slab.contour_plot(
        true_e_dat.pltpts,
        scatter_points=prepare_scatter_points(plot_last, cuh2_min),
        title=f"({_opt}, CI: {_ci}, {_k})\n Band {_index} ({_method})",
    )
    oname = f"{_method}_{_opt}_{_ci}"
    plt.savefig(f"neb_path_{oname}_{_index:04d}.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script to plot output.txt for CuH2")
    parser.add_argument(
        "--nimgs",
        help="Number of images",
        required=True,
    )
    args = parser.parse_args()

    otext = Path("output.txt").open().readlines()
    ilines = [x for x in otext if "Iteration" in x]
    iparser = re.compile(r"Iteration\s(?P<iter>\d*): (?P<coords>\[\d?.*\])")
    path_arr = np.vstack(
        [np.array(ast.literal_eval(iparser.match(x).group("coords"))) for x in ilines if iparser.match(x)]
    )
    # Ham handed approach
    n_bands = int(path_arr.shape[0] / path_arr.shape[1])
    n_imgs = int(args.nimgs)
    for idx in range(n_bands):
        _start = idx * n_imgs
        _end = _start + n_imgs
        band = path_arr[_start:_end]
        plot_band(idx, band, 5.2, "xts", "SD", "False")
