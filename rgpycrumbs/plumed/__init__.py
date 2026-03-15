# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

from rgpycrumbs.plumed.fes_calculation import calculate_fes_from_hills

__all__ = ["calculate_fes_from_hills", "find_fes_minima"]


def __getattr__(name):
    if name == "find_fes_minima":
        from rgpycrumbs.plumed.direct_reconstruction import find_fes_minima

        return find_fes_minima
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
