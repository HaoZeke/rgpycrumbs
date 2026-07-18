"""Geometry helpers: fragment detection and related utilities.

Heavy deps (ase, scipy, tblite) load only when symbols are accessed.
"""

from __future__ import annotations

from typing import Any

_LAZY_IMPORTS = {
    "DEFAULT_BOND_MULTIPLIER": "fragments",
    "DEFAULT_BOND_ORDER_THRESHOLD": "fragments",
    "DetectionMethod": "fragments",
    "build_graph_and_find_components": "fragments",
    "find_fragments_bond_order": "fragments",
    "find_fragments_geometric": "fragments",
    "merge_fragments_by_distance": "fragments",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(f"rgpycrumbs.geom.{_LAZY_IMPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
