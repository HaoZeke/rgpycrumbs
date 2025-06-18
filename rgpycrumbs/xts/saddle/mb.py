import ast
import re
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import cmcrameri.cm as cmc

from rgpycrumbs.func.muller_brown import muller_brown, muller_brown_gradient

x = np.linspace(-1.5, 1.2, 400)
y = np.linspace(-0.2, 2.0, 400)
X, Y = np.meshgrid(x, y)
Z = muller_brown([X, Y])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to plot output.txt for the Muller-Brown"
    )
    parser.add_argument(
        "--nimgs",
        help="Number of images",
        required=True,
    )
    parser.add_argument(
        "--ifile",
        help="Input file",
        default="output.txt",
    )
    args = parser.parse_args()
    nimgs = int(args.nimgs)

    otext = Path(args.ifile).open().readlines()
    ilines = [x for x in otext if "Iteration" in x]
    iparser = re.compile(r"Iteration\s(?P<iter>\d*): (?P<coords>\[\d?.*\])")
    path_arr = np.vstack(
        [
            np.array(ast.literal_eval(iparser.match(x).group("coords")))
            for x in ilines
            if iparser.match(x) and "nan" not in iparser.match(x).group("coords")
        ]
    )
    plt.figure(figsize=(12, 9))
    plt.contourf(X, Y, Z, 50, cmap=cmc.batlow, alpha=0.6)
    plt.colorbar()
    if np.any((np.sum(path_arr, axis=1) > 1e2)):
        warnings.warn("Filtered high values")
        path_arr = path_arr[np.sum(path_arr, axis=1) < 1e2]
    plt.scatter(
        path_arr[:nimgs, 0],
        path_arr[:nimgs, 1],
        marker="o",
        color="blue",
        label="Start",
        s=100,
    )
    plt.scatter(
        path_arr[-nimgs:, 0],
        path_arr[-nimgs:, 1],
        marker="*",
        color="red",
        label="End",
        s=100,
    )
    plt.scatter(
        path_arr[:, 0],
        path_arr[:, 1],
        marker="x",
        color="yellow",
        s=150,
        alpha=0.01,
        label="True Path_Arr",
    )
    plt.grid(True)
    plt.show()
