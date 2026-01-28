# Changelog

## [v0.1.0](https://github.com/HaoZeke/rgpycrumbs/tree/v0.1.0) - 2026-01-28

### Added

- feat(eon): add dictionary support ([#28ac681.eon](https://github.com/HaoZeke/rgpycrumbs/issues/28ac681.eon))
- feat(jup): add subprocess run helpers for atomistic-cookbook ([#a60da70.jup](https://github.com/HaoZeke/rgpycrumbs/issues/a60da70.jup))
- Added alignment using IRA for splitting con files.
- Added robust alignment API using Isomorphic Robust Alignment (IRA) with ASE fallback.
- Fallback to using ase minimize_rotations_and_translations if IRA fails.
- Generate hydrated logging configurations for MLFlow.
- feat(pltneb): Add a strip to handle sub-figures uniformly
- feat(pltneb): Add multiple structures and labels
- feat(pltneb): Automatically determine the smoothing factor from the global median RMSD step distance

### Developer

- Refactored alignment internals into structured dataclasses for improved API clarity.


## [v0.0.6](https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.6) - 2025-12-23

### Added

- Added a new fragment detection tool () supporting both Geometric (covalent radii) and Bond Order (GFN2-xTB) methodologies.
- feat(cli): drop dependency on click
- feat(pltneb): Add 'ira-kmax' to tweak settings for IRA

### Fixed

- Refactored internal import helper to use , fixing support for nested module imports (e.g., ).


## [v0.0.5](https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.5) - 2025-12-06

### Added

- feat(pltneb): Add 'index' rc-mode to plot against image number

### Changed

- Use white as a default background for plots
- feat(cache1D): Add disk caching for 1D profiles with rmsd using Parquet
- feat(cache2D): Add disk caching for 2D landscape plots using Parquet

### Fixed

- fix(plt_neb): Use standard spline for RMSD profiles to fix artifacts


## [v0.0.4](https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.4) - 2025-12-01

### Added

- Added `_import_from_parent_env` helper to enable fallback imports from the parent python environment.
- Updated `con_splitter` to handle multi-path trajectories via `--images-per-path` and `--path-index` arguments.

### Changed

- Migrated linting configuration to `tool.ruff.lint` and applied global formatting fixes.
- Refactored CLI architecture to support dynamic command dispatch and environment propagation for isolated scripts.
- Updated `plt_neb` to use the new import helper for optional `ira_mod` loading.

### Fixed

- Added conditional skipping for `ptmdisp` tests when `ase` is not present in the environment.
- Fixed variable name typo in `plt_neb` when falling back to globbed overlay data.


## [v0.0.3](https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.3) - 2025-10-27

### Added

- Add a FES direct reconstruction helper
- Add a version picker for the prefix package deleter
- Add an eigenvalue plotter
- Add more options for NEB visualization
- Add the PES estimated NEB plot
- Add the ruhi colorscheme

### Changed

- Use hermite interpolation for NEB plots


## [v0.0.2](https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.2) - 2025-09-06

### Added

- Helper to delete prefix packages
- Helper to generate initial paths for EON's NEB
- EON helpers using OVITO
- Enable image insets for the NEB
