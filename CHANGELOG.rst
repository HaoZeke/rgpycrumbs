rgpycrumbs [Unreleased]
=======================

Added
-----

- New ``plt-saddle`` CLI tool for dimer/saddle search trajectory visualization
  with profile, landscape, convergence, and mode-evolution plot types. Supports
  ``--ref-product`` for (initial, product) reference pairs.
- New ``plt-min`` CLI tool for minimization trajectory visualization with
  profile, landscape, and convergence plot types.
- Extended ``plt-neb`` with OCI-NEB/RONEB options: ``--mmf-peaks`` for MMF peak
  overlay (auto-detected), ``--peak-dir`` for explicit peak file directory,
  ``--show-evolution`` for band evolution across iterations.
- Four rendering backends for structure galleries: ``xyzrender`` (default,
  ball-and-stick), ``ase`` (space-filling), ``solvis`` (PyVista, transparent
  background), ``ovito`` (OVITO off-screen). All via ``--strip-renderer``.
- ``--rotation`` flag (renamed from ``--ase-rotation``) applies viewing angle
  uniformly to all rendering backends via pre-rotation of atom coordinates.
- ``--perspective-tilt`` flag for Rodrigues off-axis rotation to reveal
  atoms hidden by orthographic projection overlap. 5-10 degrees typical.
- ``--strip-spacing`` and ``--strip-dividers`` for wider gaps and vertical
  separator lines between structure images.
- NEB visualization tutorial with real Diels-Alder [4+2] cycloaddition data,
  covering all plot types and rendering backends.
- Multi-environment coverage CI workflow (``ci_coverage.yml``) combining
  test, surfaces, fragments, and eonmlflow pixi environments with
  ``--fail-under=89``.

Changed
-------

- Default rendering backend changed from ``ase`` to ``xyzrender`` for
  ball-and-stick visualization with bonds visible.

Fixed
-----

- Fixed infinite recursion in ``__init__.py`` lazy imports (replaced
  ``from rgpycrumbs import X`` with ``importlib.import_module``).
- Fixed ``chemgp/__init__.py`` re-export names after migration
  (``plot_convergence`` -> ``plot_convergence_curve``, etc.).
- Fixed ``plot_gp.py`` batch parallel I/O bug (CliRunner in threads).
- Fixed ``parsers/bless.py`` ``datetime.UTC`` for Python 3.10 compat.
- Fixed ``plt_neb.py`` dangling inline projection variables after refactor.
- Fixed ``plt_neb.py`` ``compute_profile_rmsd`` keyword-only args.


rgpycrumbs 1.4.0 (2026-03-15)
=============================

Removed
-------

- Moved ChemGP plotting (``chemgp/plotting.py``, ``chemgp/plot_gp.py``, ``chemgp/hdf5_io.py``),
  NEB CLI (``eon/plt_neb.py``), ChemGP JSONL parsers (``parsers/chemgp.py``), and PLUMED
  parsing (``plumed/fes_calculation.py``) to chemparseplot. Use ``chemparseplot.parse`` and
  ``chemparseplot.scripts`` for these modules going forward. (scope_migration)


Added
-----

- Updated ``plt-neb`` documentation with comprehensive usage examples, options table, and troubleshooting guide for landscape plotting. (docs_plt_neb)
- Tutorial documentation following Diataxis structure: quickstart, first-project
  tutorial, how-to guides (install, troubleshooting, FAQ), reference (glossary),
  and developer guides (testing, lazy imports, best practices). (docs_tutorials)
- Added 2D landscape plotting capabilities to ``plt-neb`` with gradient-enhanced surface methods (``grad_imq``, ``grad_matern``), structure strip rendering, and colorbar legend support. (feat_plt_neb_landscape)


Developer
---------

- Added ``batch`` command with ``--parallel/-j`` flag for concurrent plot generation using ThreadPoolExecutor. Provides 3-5x speedup for batch operations with 4-8 workers. Pattern adopted from nebmmf_repro project. (perf_batch)
- Added ``safe_plot()`` decorator and ``validate_hdf5_structure()`` function for robust error handling. Better error messages for missing files and invalid structures. (perf_errorhandling)
- Added complete type hints to chemgp modules (hdf5_io.py, plotting.py) reaching 90%+ coverage. Improves IDE support and code documentation. (perf_typehints)


Changed
-------

- Changed default surface method in ``plt-neb`` from ``rbf`` to ``grad_imq`` for higher-quality gradient-enhanced interpolation. (feat_plt_neb_defaults)


rgpycrumbs v1.3.0 (2026-03-09)
==============================

Added
-----

- ``[all]`` optional dependency extra combining plot, eon, and other extras. (all-extra)
- ASV benchmark infrastructure with asv-perch CI integration. (asv-benchmarks)
- ChemGP JSONL parsers for optimization logs, convergence data, and GP diagnostics (``rgpycrumbs.parsers.chemgp``). (chemgp-parsers)
- Dynamic dependency resolution via ``ensure_import`` for optional packages. (ensure-import)
- ``plt-gp`` PEP 723 CLI script for ChemGP figure generation with batch mode and landscape subcommand. (plt-gp-cli)


Developer
---------

- ASV benchmark infrastructure, prek linting, taplo formatting, codespell configuration. (dev-infra)


Changed
-------

- Documentation overhaul: Shibuya theme, quickstart page, Google-style docstrings, CI/PyPI badges, and README generation from Org source. (docs-overhaul)


Fixed
-----

- Skip missing H5 inputs in batch mode instead of failing. (batch-skip)
- Fix ``plt-neb`` keyword arguments for ``compute_profile_rmsd`` call. (plt-neb-kwargs)
- CI formatting and py3.10 compatibility fixes. (ci-py310)


rgpycrumbs v1.2.0 (2026-02-24)
==============================

Added
-----

- Added --strip-renderer CLI option for selecting structure rendering backend. (#1)
- Export Nystrom threshold, default inducing count, and ``nystrom_paths_needed``
  from ``rgpycrumbs.surfaces`` so callers can trim data before expensive
  RMSD/IRA aggregation. (#2)
- Added --source traj support to plt_neb for loading NEB paths from ASE-readable trajectory files (extxyz, .traj, etc.). (#3)
- Added --source hdf5 support to plt_neb for loading NEB paths from ChemGP HDF5 output files. (#4)
- Added --n-inducing flag to eON surface-fitting CLI for controlling the Nystrom approximation inducing-point count. (#5)


Developer
---------

- Added versionadded directives to all public API docstrings. (#7)


Changed
-------

- Per-image RMSD alignment in rgpycrumbs.geom now runs in parallel over threads for improved performance on large NEB paths. (#6)


rgpycrumbs v1.1.0 (2026-02-21)
==============================

Added
-----

- Comprehensive pytest suite for surface models, covering fit, prediction, variance, and optimization.


Developer
---------

- Added detailed API docstrings and Org-mode documentation for surfaces, including uncertainty and variance windowing.


Changed
-------

- Refactored rgpycrumbs.surfaces into a structured package with BaseSurface and BaseGradientSurface abstractions.


rgpycrumbs v1.0.1 (2026-02-17)
==============================

Added
-----

- CI workflow to auto-generate ``README.md`` from ``readme_src.org`` on push to main.
- Downstream users page (``used_by.org``) listing public projects that depend on ``rgpycrumbs``.


Changed
-------

- Updated ``readme_src.org`` with current ``uv`` + ``hatch-vcs`` + ``twine`` development and release workflow.


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
