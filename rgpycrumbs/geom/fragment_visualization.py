"""
PyVista-based visualization for molecular fragment analysis.

Renders molecular systems with scalar-coded bond orders using CPK coloring
and the batlow colormap from cmcrameri.

.. versionadded:: 0.0.6
"""

import logging

import cmcrameri.cm as cmcrameri_cm
import matplotlib as mpl
import numpy as np
import pyvista as pv
from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.neighborlist import build_neighbor_list, natural_cutoffs

from rgpycrumbs.geom.fragments import DetectionMethod

mpl.colormaps.register(cmcrameri_cm.batlow, force=True)
cmap_name = "batlow"

# Plot Settings
SCALAR_BAR_ARGS = {
    "title": "Wiberg Bond Order",
    "vertical": True,
    "position_x": 0.85,  # Slightly away from the right edge
    "position_y": 0.05,  # Start near the bottom
    "height": 0.9,  # Stretch to cover 90% of the window height
    "width": 0.05,  # Adjust thickness as needed
    "title_font_size": 20,
    "label_font_size": 16,
}

MIN_DIST_ATM = 1e-4


def visualize_with_pyvista(
    atoms: Atoms,
    method: DetectionMethod,
    bond_data: float | np.ndarray,
    nonbond_cutoff: float = 0.05,
    bond_threshold: float = 0.8,
    radius_type: str = "natural",
) -> None:
    """Renders the molecular system with scalar-coded bond orders.

    .. versionadded:: 0.0.6
    """
    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.set_background("white")

    # CPK Colors
    cpk_colors = {
        1: "#FFFFFF",
        6: "#b5b5b5",
        7: "#0000FF",
        8: "#FF0000",
        9: "#90E050",
        15: "#FF8000",
        16: "#FFFF00",
        17: "#00FF00",
        35: "#A62929",
        53: "#940094",
    }
    default_color = "#FFC0CB"

    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()
    radii = covalent_radii[nums] * 0.45

    # Render Atoms
    for i, (p, n) in enumerate(zip(pos, nums, strict=False)):
        sphere = pv.Sphere(
            radius=radii[i], center=p, theta_resolution=24, phi_resolution=24
        )
        plotter.add_mesh(
            sphere,
            color=cpk_colors.get(n, default_color),
            specular=0.5,
            smooth_shading=True,
        )

    # Render Bonds based on Method
    if method == DetectionMethod.GEOMETRIC:
        multiplier = float(bond_data)
        if radius_type == "covalent":
            cutoffs = covalent_radii[atoms.get_atomic_numbers()] * multiplier
        else:
            cutoffs = natural_cutoffs(atoms, mult=multiplier)
        nl = build_neighbor_list(atoms, cutoffs=cutoffs, self_interaction=False)

        for i in range(len(atoms)):
            indices, _ = nl.get_neighbors(i)
            for j in indices:
                if i < j:
                    p1, p2 = pos[i], pos[j]
                    cyl = pv.Cylinder(
                        center=(p1 + p2) / 2,
                        direction=p2 - p1,
                        radius=0.15,
                        height=np.linalg.norm(p2 - p1),
                    )
                    plotter.add_mesh(cyl, color="darkgrey", specular=0.2)

    elif method == DetectionMethod.BOND_ORDER:
        matrix = bond_data
        # Ensure matrix is a numpy array
        matrix = np.asarray(matrix)

        # Identify pairs above threshold
        indices = np.argwhere(np.triu(matrix, k=1) > nonbond_cutoff)

        if indices.size == 0:
            logging.warning("No interactions found above cutoff.")
            plotter.show()
            return

        visible_wbo = matrix[indices[:, 0], indices[:, 1]]
        min_bo, max_bo = visible_wbo.min(), visible_wbo.max()

        # Avoid division by zero if all bond orders are equal
        bo_range = max_bo - min_bo if max_bo > min_bo else 1.0

        bonded_meshes = []
        weak_meshes = []

        for idx_pair in indices:
            i, j = idx_pair
            wbo = matrix[i, j]
            p1, p2 = pos[i], pos[j]
            vec = p2 - p1
            dist = np.linalg.norm(vec)
            # Skip overlapping atoms
            if dist < MIN_DIST_ATM:
                continue

            if wbo >= bond_threshold:
                # Normalize radius: stronger bonds appear thicker
                norm_bo = np.clip((wbo - min_bo) / bo_range, 0.0, 1.0)
                radius = 0.08 + (0.01 * norm_bo)

                cyl = pv.Cylinder(
                    center=(p1 + p2) / 2,
                    direction=vec,
                    radius=radius,
                    height=dist,
                    resolution=15,
                )
                # Assign scalar to points for smoother rendering
                cyl.point_data["WBO"] = np.full(cyl.n_points, wbo)
                bonded_meshes.append(cyl)
            else:
                # Weak interaction dots
                n_dots = max(2, int(dist / 0.2))
                for k in range(n_dots + 1):
                    dot_pos = p1 + (k / n_dots) * vec
                    dot = pv.Sphere(radius=0.04, center=dot_pos)
                    dot.point_data["WBO"] = np.full(dot.n_points, wbo)
                    weak_meshes.append(dot)

        # Merge and Add to Plotter
        if bonded_meshes:
            plotter.add_mesh(
                pv.merge(bonded_meshes),
                scalars="WBO",
                cmap="batlow",
                clim=[min_bo, max_bo],
                smooth_shading=True,
                scalar_bar_args=SCALAR_BAR_ARGS,
            )

        if weak_meshes:
            plotter.add_mesh(
                pv.merge(weak_meshes),
                scalars="WBO",
                cmap="batlow",
                clim=[min_bo, max_bo],
                opacity=0.6,
                show_scalar_bar=False,
            )

    logging.info("Opening visualization...")
    plotter.show()
