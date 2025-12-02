## [v0.0.4](https://github.com/theochemui/eongit/tree/v0.0.4) - 2025-12-01

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


## [v0.0.3](https://github.com/theochemui/eongit/tree/v0.0.3) - 2025-10-27

### Added

- Add a FES direct reconstruction helper
- Add a version picker for the prefix package deleter
- Add an eigenvalue plotter
- Add more options for NEB visualization
- Add the PES estimated NEB plot
- Add the ruhi colorscheme

### Changed

- Use hermite interpolation for NEB plots


## [v0.0.2](https://github.com/theochemui/eongit/tree/v0.0.2) - 2025-09-06

### Added

- Helper to delete prefix packages
- Helper to generate initial paths for EON's NEB
- EON helpers using OVITO
- Enable image insets for the NEB
