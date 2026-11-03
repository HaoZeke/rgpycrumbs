import ast
import re
import warnings
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np

from rgpycrumbs.func.muller_brown import muller_brown

ITERATION_LINE = re.compile(r"Iteration\s(?P<iter>\d*): (?P<coords>\[\d?.*\])")


def _surface_grid():
    x = np.linspace(-1.5, 1.2, 400)
    y = np.linspace(-0.2, 2.0, 400)
    X, Y = np.meshgrid(x, y)
    Z = muller_brown([X, Y])
    return X, Y, Z


def parse_iteration_paths(lines):
    """Stack the coordinates of the ``Iteration`` lines of an optimizer log.

    Lines whose coordinates contain ``nan`` are skipped.

    Raises:
        ValueError: If no ``Iteration`` line is left to stack.

    .. versionadded:: 1.14.0
    """
    coords = []
    for line in lines:
        match = ITERATION_LINE.match(line)
        if match and "nan" not in match["coords"]:
            coords.append(np.array(ast.literal_eval(match["coords"])))
    if not coords:
        raise ValueError("the log has no Iteration line without nan coordinates")
    return np.vstack(coords)


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
    X, Y, Z = _surface_grid()

    path_arr = parse_iteration_paths(Path(args.ifile).read_text().splitlines())
    plt.figure(figsize=(12, 9))
    plt.contourf(X, Y, Z, 50, cmap=cmc.batlow, alpha=0.6)
    plt.colorbar()
    if np.any(np.sum(path_arr, axis=1) > 1e2):
        warnings.warn("Filtered high values", stacklevel=2)
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
