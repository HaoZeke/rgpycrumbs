rgpycrumbs 1.9.23 (2026-07-15)
==============================

Documentation
-------------

- Full orgmode pass: new ``tools/eon/plot_config.org``; ``plt-min`` /
  ``plt-neb`` / minimization tutorial / FAQ / troubleshooting / glossary
  document TOML-only ``auto_thin`` (default off) and ``--config`` workflow.
- ``--config`` help text lists surface-fit keys and min/neb examples.

rgpycrumbs 1.9.22 (2026-07-15)
==============================

Changed
-------

- Surface-fit knobs ``auto_thin`` / ``max_surface_points`` are **TOML-only**
  (``[shared]`` or ``[min]`` / ``[neb]`` via ``--config``), not CLI flags.
  Defaults remain off / 64. Example in ``eon/examples/plot_config.example.toml``.
- ``plt-neb`` and ``plt-min`` forward those keys to chemparseplot>=1.9.10
  ``SurfaceFitConfig`` / ``plot_landscape_surface``.

rgpycrumbs 1.9.21 (2026-07-15)
==============================

Changed
-------

- Floor ``chemparseplot[neb,plot]`` to ``>=1.9.9`` in PEP 723 scripts and
  ``ensure_import`` pins.
- Plot TOML keys for surface fit (defaults off); see 1.9.22 for TOML-only
  policy.

rgpycrumbs 1.9.20 (2026-07-10)
==============================

Fixed
-----

- Profile panels with crit_points use a content-sized structure strip under the axes (re-rendered after layout) instead of crushed insets on the energy curve. (plt-neb-profile-strip)


rgpycrumbs 1.9.19 (2026-07-09)
==============================

Added
-----

- Docs: consumer path documents ``chemparseplot.api.extract_orca_geomscan_energy``
  as the stable parse→typed library path (with ``rgpycrumbs.api`` for config/pins).

rgpycrumbs 1.9.18 (2026-07-09)
==============================

Added
-----

- Stable public suite surface ``rgpycrumbs.api`` (config, locks, ensure_import)
  for consumers (chemparseplot, pychum, wailord). Documented in
  ``docs/orgmode/explanation/public_api.org``.

rgpycrumbs 1.9.17 (2026-07-09)
==============================

Added
-----

- ``rgpycrumbs config show`` prints merged suite config (sources, lock path,
  force_uv, package pins). Suite architecture notes under
  ``docs/orgmode/explanation/suite_architecture.org``.

rgpycrumbs 1.9.16 (2026-07-09)
==============================

Changed
-------

- Config identity is **ecosystem-wide ``rgpkgs``**, not package-private:
  ``~/.config/rgpkgs/config.toml`` and project ``rgpkgs.toml`` (shared with
  chemparseplot and other suite tools). Legacy ``rgpycrumbs.toml`` /
  ``~/.config/rgpycrumbs/`` still load as fallbacks. Env aliases:
  ``RGPKGS_LOCK``, ``RGPKGS_CONFIG``, ``RGPKGS_FORCE_UV``, ``RGPKGS_AUTO_DEPS``
  (``RGPYCRUMBS_*`` kept).

rgpycrumbs 1.9.15 (2026-07-09)
==============================

Added
-----

- Layered **TOML config** for pins and dispatch defaults:
  ``~/.config/rgpycrumbs/config.toml`` (global), project ``rgpycrumbs.toml`` /
  ``.rgpycrumbs.toml`` (walk-up from CWD), optional ``RGPYCRUMBS_CONFIG``.
  Sections: ``[dispatch]`` (``auto_deps``, ``force_uv``), ``[pins]`` (``lock``),
  ``[pins.packages]`` (explicit overrides). Precedence: CLI > env > project >
  global. Example: ``docs/examples/rgpycrumbs.config.toml``.

rgpycrumbs 1.9.14 (2026-07-09)
==============================

Added
-----

- Optional lock consume is not SBOM-only: ``--lock`` / ``RGPYCRUMBS_LOCK`` accepts
  **uv.lock**, **PEP 751** ``pylock.toml`` / ``pylock.*.toml``, and **CycloneDX**
  JSON. Pins feed ``uv run --constraints`` / ``UV_CONSTRAINT`` and
  ``ensure_import`` (``RGPYCRUMBS_LOCK_PINS``). ``--sbom`` / ``RGPYCRUMBS_SBOM``
  remain aliases. Auto-detect by path/content; non-PyPI CDX rows still skipped.

rgpycrumbs 1.9.13 (2026-07-08)
==============================

Added
-----

- Optional **CycloneDX SBOM consume** on the CLI: ``--sbom PATH`` or
  ``RGPYCRUMBS_SBOM``. PyPI components (``pkg:pypi/...``) become install pins for
  ``uv run --constraints`` and ``ensure_import`` / AUTO_DEPS. Non-PyPI entries
  (e.g. eb-stack ``pkg:generic/...``) are skipped. Missing path fails clearly;
  unset SBOM leaves default floating resolution.
- PEP 723 headers declare ``rgpycrumbs>=1.9.13`` so standalone ``uv run`` of a
  plot script can resolve the package (dispatcher remains preferred entry).

rgpycrumbs 1.9.12 (2026-07-08)
==============================

Removed
-------

- Runtime feature extras ``[analysis]``, ``[surfaces]``, ``[interpolation]``,
  and ``[all]``. They diluted the real design: CLI deps via PEP 723 + ``uv``,
  library deps via ``ensure_import`` + ``RGPYCRUMBS_AUTO_DEPS=1``. Dev extras
  (``test`` / ``lint`` / ``release``) remain.
- ``_DEPENDENCY_MAP`` no longer stores fake extra names; values are pip specs
  only, and error messages say ``pip install "<spec>"`` or enable AUTO_DEPS.

rgpycrumbs 1.9.11 (2026-07-08)
==============================

Fixed
-----

- Restore auto-deps design: ``[analysis]`` does **not** hard-depend on
  ``jax`` / ``adjustText``. Those stay optional and resolve via uv PEP 723
  (isolated dispatch) or ``ensure_import`` (in-env).
- CLI dispatch now defaults ``RGPYCRUMBS_AUTO_DEPS=1`` so in-env ``plt-*``
  can cache-install heavies without host env pins. Set ``=0`` to disable.
- ``plt-neb`` loads ``adjustText`` via ``ensure_import`` (not a bare import).

rgpycrumbs 1.9.10 (2026-07-08)
==============================

Changed
-------

- PEP 723 pins for plot scripts: readcon>=0.13.1, chemparseplot[neb,plot].
- (Yanked design) briefly hard-pinned jax/adjustText on ``[analysis]``;
  corrected in 1.9.11 — use auto-deps / uv instead.

rgpycrumbs 1.9.9 (2026-07-07)
=============================

Fixed
-----

- ``plt-min`` landscapes: title from job label (e.g. ``Reactant minimization``),
  ``initial`` / ``minimized`` endpoint and strip labels, relative-energy colorbar,
  and readable axis labels (via chemparseplot 1.9.7 single-ended pipeline).

rgpycrumbs 1.9.8 (2026-07-07)
=============================

Fixed
-----

- Landscape structure strip uses **two rows** (``max_cols=6``) with a taller
  strip panel and higher zoom so molecules are larger and readable.
- Keep square 1:1 (s, d) panel: ``Δs = Δd`` (e.g. both 1.34 Å); d ticks show
  ``±half`` while s shows ``0→span`` — same physical length on both axes.

rgpycrumbs 1.9.7 (2026-07-07)
=============================

Fixed
-----

- ``plt-neb`` projected landscapes use a **square** equal-aspect panel with
  ``Δd = Δs`` so 1 Å of progress matches 1 Å of orthogonal deviation (true 1:1
  RMSD metric). Figure is still sized to map+strip (no left void); structure
  strip is re-rendered after layout.

rgpycrumbs 1.9.6 (2026-07-07)
=============================

Fixed
-----

- ``plt-neb`` projected landscape layout: size the figure to map+strip (no left
  void in a leftover 12×8 cell), re-render the structure strip *after* layout so
  molecules use the final large axes, keep equal aspect for (s, d).

rgpycrumbs 1.9.5 (2026-07-07)
=============================

Fixed
-----

- ``plt-neb`` projected landscapes always use equal aspect (``s`` and ``d`` are
  both Å from the same RMSD metric). Figure size follows the data aspect so the
  map is not stretched; colorbar/strip are placed against that box. Never use
  ``aspect='auto'`` for (s, d).

rgpycrumbs 1.9.4 (2026-07-07)
=============================

Fixed
-----

- First pass at equal-aspect (s, d) landscapes (superseded by 1.9.5 layout).

rgpycrumbs 1.9.3 (2026-07-07)
=============================

Fixed
-----

- ``plt-neb`` 1D profiles mark the saddle with a gold star, ``SP`` label box, and
  a vertical guide line (was a small white star easy to miss).
- ``plt-neb`` 2D landscapes use path-driven ``d`` limits and tighter save padding
  so GP-masked margins no longer dominate the frame.

rgpycrumbs 1.9.0 (2026-07-07)
=============================

Added
-----

- Prefer the active interpreter for PEP 723 script dispatch when the readcon/plot
  stack is already importable (avoids broken isolated ``uv run`` Python ABIs when
  plotting eOn 2.16 ``con_spec_version=2`` outputs). Override with
  ``RGPYCRUMBS_FORCE_UV=1``; force in-env with ``RGPYCRUMBS_DEV=1`` or ``--dev``.
- Declarative TOML plot config for NEB/min/saddle CLIs; CON trajectory splitter
  via readcon.

Changed
-------

- Analysis extras require ``readcon>=0.13.1`` and ``chemparseplot[neb,plot]>=1.8.0``.


rgpycrumbs 1.8.0 (2026-06-27)
=============================

Added
-----

- ``plt-neb-stitch`` CLI for full-path 1D strip and 2D landscape plots stitched from NEB segments. (#1800)
- ``gen-dimer`` seeds dimer saddle searches from all NEB peaks; ``plt-kmc`` plots N2-ejection KMC timelines. (#1801)
- Profile strip rendering for NEB plots via chemparseplot/xyzrender overlays. (#1802)
- Energy unit selection for NEB (``plt-neb``) and single-ended (``plt-min`` / ``plt-saddle``) landscape tools. (#1803)
- ``plt-min`` ``--energy-cap`` / ``--energy-cap-window`` options to limit landscape energy scale. (#1804)


Developer
---------

- Cocogitto (``cog``) release tooling and contributor release docs; CI creates GitHub Release for Zenodo on publish. (#1813)
- Tests fail loudly on broken optional imports; pytest-pep723 required in test env; tighter CLI plotting gates. (#1814)


Changed
-------

- eOn plot scripts consume typed chemparseplot NEB/single-ended helpers and shared renderer option bundles. (#1812)
- CLI warns when PEP 723 scripts are imported directly instead of through the dispatcher. (#1815)


Fixed
-----

- NEB xyzrender strip layout budgets, clearance, and sizing for readable profile overlays. (#1805)
- Single-ended landscape overlays stay on the shared projection basis; ``plt-min`` default prefix aligned. (#1806)
- Dispatcher preserves parent site-packages paths; PEP 723 plot scripts execute when invoked via the CLI. (#1807)
- Lazy imports for ``xts`` / ``cuh2`` so optional plotting stacks do not break base imports. (#1808)
- ``ira_kmax`` default 14, Init/Min label boxes, larger SP star; NEB viewport includes trajectory d-range. (#1809)
- Require chemparseplot eOn APIs for NEB overlays; simplify OCI overlays and local dispatch. (#1810)
- PLUMED direct reconstruction script metadata aligned with dispatcher expectations. (#1811)


rgpycrumbs 1.7.0 (2026-04-07)
=============================

Added
-----

- Integrated pytest-pep723 plugin for static validation of PEP 723 inline script dependencies in CI.


Developer
---------

- Added OIDC trusted publishing workflow for PyPI releases, gated on tests, lints, and docs.
- Migrated documentation deployment from GitHub Pages to Cloudflare Pages.


Fixed
-----

- Added missing numpy and rich to PEP 723 inline script dependencies in plot_gp.py.
- Added readcon to PEP 723 inline script dependencies in plt_neb.py and plt_saddle.py, fixing dispatch failures.


rgpycrumbs 1.5.0 (2026-03-23)
=============================

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
