rgpycrumbs v1.0.0 (2026-02-17)
==============================

Removed
-------

- ``ira_mod`` from pip-installable optional dependencies (conda-only; managed exclusively via ``pixi``). (#37)
- ``pdm`` and ``pip`` from ``pixi`` base dependencies. (#37)


Added
-----

- Library modules extracted from chemparseplot: ``basetypes`` (NEB, dimer, saddle point data types), ``surfaces`` (JAX-based GP surface fitting with gradient-enhanced kernels), ``interpolation`` (spline helpers), and ``parsers`` (common parsing patterns). (#36)
- Public API with lazy loading via ``__getattr__`` for ``surfaces``, ``basetypes``, and ``interpolation`` modules. (#36)
- Click-based CLI dispatcher with ``rgpycrumbs`` entry point. (#36)
- Optional dependency groups for ``surfaces`` (JAX), ``analysis`` (ASE, scipy), and ``interpolation`` (scipy). (#36)
- ``test`` and ``lint`` optional dependency groups in ``pyproject.toml``. (#37)
- ``rich`` as a core dependency (used across CLI modules). (#37)


Changed
-------

- ``uv``-first development workflow; ``pixi`` now only needed for conda-gated features (IRA, fragments). (#37)
- CI split into ``uv`` job (pure, interpolation, align tests on Python 3.10-3.12) and ``pixi`` job (fragments, surfaces tests requiring conda packages). (#37)


Fixed
-----

- Interpolation tests correctly marked as ``interpolation`` (needs scipy), not ``pure``. (#37)


rgpycrumbs v0.1.2 (2026-01-28)
==============================

Fixed
-----

- Additional imports


rgpycrumbs v0.1.1 (2026-01-28)
==============================

Fixed
-----

- Add an __init__ for modules
- Rework documentation


Changelog
=========

`v0.1.0 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.1.0>`__ - 2026-01-28
---------------------------------------------------------------------------

Added
~~~~~

-  feat(eon): add dictionary support
   (`#28ac681.eon <https://github.com/HaoZeke/rgpycrumbs/issues/28ac681.eon>`__)
-  feat(jup): add subprocess run helpers for atomistic-cookbook
   (`#a60da70.jup <https://github.com/HaoZeke/rgpycrumbs/issues/a60da70.jup>`__)
-  Added alignment using IRA for splitting con files.
-  Added robust alignment API using Isomorphic Robust Alignment (IRA)
   with ASE fallback.
-  Fallback to using ase minimize_rotations_and_translations if IRA
   fails.
-  Generate hydrated logging configurations for MLFlow.
-  feat(pltneb): Add a strip to handle sub-figures uniformly
-  feat(pltneb): Add multiple structures and labels
-  feat(pltneb): Automatically determine the smoothing factor from the
   global median RMSD step distance

Developer
~~~~~~~~~

-  Refactored alignment internals into structured dataclasses for
   improved API clarity.

`v0.0.6 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.6>`__ - 2025-12-23
---------------------------------------------------------------------------

.. _added-1:

Added
~~~~~

-  Added a new fragment detection tool () supporting both Geometric
   (covalent radii) and Bond Order (GFN2-xTB) methodologies.
-  feat(cli): drop dependency on click
-  feat(pltneb): Add ‘ira-kmax’ to tweak settings for IRA

Fixed
~~~~~

-  Refactored internal import helper to use , fixing support for nested
   module imports (e.g., ).

`v0.0.5 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.5>`__ - 2025-12-06
---------------------------------------------------------------------------

.. _added-2:

Added
~~~~~

-  feat(pltneb): Add ‘index’ rc-mode to plot against image number

Changed
~~~~~~~

-  Use white as a default background for plots
-  feat(cache1D): Add disk caching for 1D profiles with rmsd using
   Parquet
-  feat(cache2D): Add disk caching for 2D landscape plots using Parquet

.. _fixed-1:

Fixed
~~~~~

-  fix(plt_neb): Use standard spline for RMSD profiles to fix artifacts

`v0.0.4 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.4>`__ - 2025-12-01
---------------------------------------------------------------------------

.. _added-3:

Added
~~~~~

-  Added ``_import_from_parent_env`` helper to enable fallback imports
   from the parent python environment.
-  Updated ``con_splitter`` to handle multi-path trajectories via
   ``--images-per-path`` and ``--path-index`` arguments.

.. _changed-1:

Changed
~~~~~~~

-  Migrated linting configuration to ``tool.ruff.lint`` and applied
   global formatting fixes.
-  Refactored CLI architecture to support dynamic command dispatch and
   environment propagation for isolated scripts.
-  Updated ``plt_neb`` to use the new import helper for optional
   ``ira_mod`` loading.

.. _fixed-2:

Fixed
~~~~~

-  Added conditional skipping for ``ptmdisp`` tests when ``ase`` is not
   present in the environment.
-  Fixed variable name typo in ``plt_neb`` when falling back to globbed
   overlay data.

`v0.0.3 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.3>`__ - 2025-10-27
---------------------------------------------------------------------------

.. _added-4:

Added
~~~~~

-  Add a FES direct reconstruction helper
-  Add a version picker for the prefix package deleter
-  Add an eigenvalue plotter
-  Add more options for NEB visualization
-  Add the PES estimated NEB plot
-  Add the ruhi colorscheme

.. _changed-2:

Changed
~~~~~~~

-  Use hermite interpolation for NEB plots

`v0.0.2 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.2>`__ - 2025-09-06
---------------------------------------------------------------------------

.. _added-5:

Added
~~~~~

-  Helper to delete prefix packages
-  Helper to generate initial paths for EON’s NEB
-  EON helpers using OVITO
-  Enable image insets for the NEB
