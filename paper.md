---
title: "`rgpycrumbs`: Post-processing tools for saddle point searches and reaction path analysis"
tags:
  - Python
  - computational chemistry
  - Gaussian processes
  - nudged elastic band
  - reaction kinetics
  - saddle point search
  - fragment detection
authors:
  - name: Rohit Goswami
    orcid: 0000-0002-2393-8056
    affiliation: "1, 2"
affiliations:
  - name: Science Institute and Faculty of Physical Sciences, University of Iceland, 107 Reykjavik, Iceland
    index: 1
  - name: "EPFL - Ecole Polytechnique Federale de Lausanne, Switzerland"
    index: 2
date: 10 October 2026
bibliography: paper.bib
---

# Summary

Saddle point searches and minimum energy path calculations are the
rate-limiting step in computational studies of chemical kinetics. Once
converged, the raw output (positions, energies, forces, eigenvalues at each
image) requires substantial post-processing before it can be interpreted:
energy profiles must be interpolated with force consistency, paths must be
projected into low-dimensional coordinate systems, chemical transformations at
the transition state must be identified, and sparse landscape data must be
interpolated onto grids for visualization. `rgpycrumbs` provides the
computational kernels for these tasks (surface fitting, alignment,
force-consistent interpolation) together with a PEP 723-based CLI dispatcher
that runs analysis scripts under `uv` in isolated environments so heavy
optional dependencies (JAX, ASE, `readcon`, plotting stacks) do not pollute a
minimal install. Format-specific parsing, unit-aware NEB and single-ended
figures, multi-segment NEB stitching, and metadata-native eOn `.con` energies
live in the companion `chemparseplot` package [@chemparseplot]; input
generation for ORCA and eOn is handled by `pychum`.

# Statement of need

A converged saddle point search produces a transition state geometry and the
minimum energy path connecting it to adjacent minima. Three questions follow
immediately:

1.  **What is the energy profile along the path?** The standard representation
    plots energy $E$ against a reaction coordinate $s$ (cumulative path length
    or RMSD). Hermite interpolation using the parallel force component
    $f_\parallel^{(i)} = \mathbf{F}_i \cdot \hat{\boldsymbol{\tau}}_i$ as
    derivative data [@henkelman2000improved] produces physically consistent
    profiles; unconstrained splines through energy values alone oscillate.

2.  **What does the energy landscape look like around the path?** A 2D
    representation in coordinates $(r, p)$, the RMSD from reactant and
    product structures, requires aligning each image to the endpoints
    (handling permutational isomers via IRA [@gunde2021ira] when needed),
    computing synthetic gradients by projecting $f_\parallel$ onto the
    $(r, p)$ tangent direction, and fitting the sparse data with kernel
    interpolation [@goswami2026rmsd]. Gradient-enhanced kernels produce
    smoother surfaces from fewer points than standard RBF methods.

3.  **What chemical transformation occurs at the saddle point?** Bond breaking
    and formation events are identified by computing Wiberg Bond Orders (WBO)
    [@wiberg1968] from a GFN2-xTB [@bannwarth2019gfn2] single-point
    calculation at the transition state and comparing the bond order matrix
    to the reactant. Geometric fragment detection (scaled covalent radii)
    provides a fast alternative when electronic structure is not available.

These operations recur across saddle point search codes (eOn, ORCA, ASE,
ChemGP) and file formats (.dat/.con, extxyz, HDF5), but existing
implementations are one-off scripts coupled to specific workflows.
`rgpycrumbs` consolidates the computation into tested, importable modules;
`chemparseplot` handles format-specific parsing and delegates heavy
computation to `rgpycrumbs`. The intended users are computational chemists
and physicists working on reaction rates, saddle point searches and
long-timescale molecular dynamics.

The algorithms originate from a C++ implementation (`gpr_optim`
[@gpr_optim]), a port of the MATLAB code by Koistinen et al.
[@koistinen2017; @koistinen2019], developed during the doctoral work
[@goswami2025thesis]. The Python library makes these methods accessible
without C++ compilation.

# State of the field

ASE [@larsen2017ase] provides atomic simulation infrastructure and NEB
implementations but not 2D landscape interpolation, gradient-enhanced kernel
fitting, or bond-order-based fragment detection at transition states.
pymatgen [@ong2013pymatgen] targets materials science database workflows with
different scope. CatLearn implements GP-accelerated NEB within ASE
[@torres2019mlneb] but focuses on the optimization loop, not post-processing.
sGDML [@chmiela2017sgdml] fits kernel-based molecular force fields for
dynamics rather than reaction path analysis. The `tblite` package provides the
GFN2-xTB [@bannwarth2019gfn2] backend for Wiberg Bond Order computation but
does not itself perform fragment detection or transition state analysis.
`rgpycrumbs` uses ASE `Atoms` objects as its native data type and complements
these tools rather than replacing them.

`rgpycrumbs` occupies the space between a converged calculation and its
chemical interpretation: projecting paths into readable coordinates, fitting
landscapes from sparse data using gradient information, and determining which
bonds break at the saddle point.

# Software design

The package separates into library modules and a CLI dispatcher.

**Surface fitting** (`rgpycrumbs.surfaces`). JAX-based [@jax2018github]
Gaussian process regression. Standard surfaces use thin-plate spline,
Matern 3/2 and inverse multiquadric kernels; gradient-enhanced variants that
use forces as well as energies cover the Matern 3/2, inverse multiquadric,
rational quadratic and squared exponential kernels. A Nystrom approximation
handles larger datasets. These interpolate 2D energy or eigenvalue landscapes
from sparse NEB data in the $(r, p)$ coordinate system.

**Structural analysis** (`rgpycrumbs.geom`). Three capabilities: (i) distance
and bond matrix construction from ASE `Atoms` objects, with fragment detection
via connected components of the bond graph; (ii) Wiberg Bond Order analysis
through GFN2-xTB (`tblite`), which identifies which bonds break or form at the
transition state, which is the information a chemist needs beyond the
barrier height; (iii) IRA alignment for RMSD calculation between structures that may
be permutational isomers, which is required for well-defined $(r, p)$
coordinates when the reaction involves atomic exchange.

**Interpolation** (`rgpycrumbs.interpolation`). Hermite spline interpolation
using both energy and force data for 1D profiles along reaction coordinates.

**Data types** (`rgpycrumbs.basetypes`). Shared structures for NEB paths,
saddle point measures, and molecular geometries.

**PLUMED integration** (`rgpycrumbs.plumed`). A CLI for free energy surface
reconstruction from metadynamics simulations. HILLS file parsing and FES
kernel summation are provided by `chemparseplot`; `rgpycrumbs` retains the
workflow that chains parsing, minima detection, and visualization into a
single script.

**Test potentials** (`rgpycrumbs.func`). The Muller-Brown surface for
algorithm validation.

## On-demand dependency resolution

Research workflows in computational chemistry require tools with mutually
exclusive binary dependencies. OVITO (crystal defect analysis) and tblite
(tight-binding electronic structure) cannot coexist in a single Python
environment; JAX and PyTorch builds may also conflict. The package addresses
this through two mechanisms that share one design: a lightweight core with
dependencies resolved on demand.

For CLI scripts, the `rgpycrumbs.cli` dispatcher invokes each script in an
isolated subprocess via `uv`, using PEP 723 inline metadata to declare
per-script dependencies. The fragment detection script, for instance, declares
`ase` and `pyvista` among its dependencies and runs in a fresh environment
without polluting the host. A lock file or SBOM (`uv.lock`, PEP 751
`pylock.toml` or CycloneDX JSON) passed with `--lock` or `RGPYCRUMBS_LOCK`
pins those dependencies, so the same environment can be rebuilt for a later
run.

For library modules, `ensure_import` resolves dependencies at first use
through a priority chain: current environment, parent environment fallback,
XDG cache lookup, and (when opted in via `RGPYCRUMBS_AUTO_DEPS=1`) automatic
installation via `uv pip install --target` into a persistent cache directory.
The resolver detects CUDA availability and selects CPU-only package variants
when no GPU is present. A plain `pip install rgpycrumbs` therefore provides
the full import surface; heavy dependencies materialize only when first
accessed.

The correctness of the PEP 723 metadata is enforced by `pytest-pep723`
[@pytest_pep723], a pytest plugin developed alongside `rgpycrumbs` that
statically verifies every import in a dispatched script is declared in its
inline metadata block. It parses the `# /// script` dependencies, extracts all
import statements via Python's AST, and reports uncovered imports. This
catches a class of bug where a developer adds a new import but forgets to
update the inline metadata, a failure that only manifests at dispatch time
in a clean environment. The plugin runs in CI on every push.

## Companion libraries

The suite is three packages with a fixed role split:

- `rgpycrumbs`: computational kernels and the PEP 723 / `uv` CLI
  dispatcher that orchestrates eOn visualization scripts (`plt-neb`,
  `plt-neb-stitch`, `plt-min`, `plt-saddle`, `gen-dimer`, `plt-kmc`). Plot
  CLIs orchestrate I/O and call `chemparseplot` plotting APIs rather than
  reimplementing NEB renders.
- `chemparseplot`: parsers (eOn via `readcon`, ORCA, ChemGP, PLUMED, ASE
  trajectories), unit-aware plotting, and stitching for multi-segment NEB
  bands. It does not implement heavy numerics.
- `pychum`: input generation only (ORCA and eOn templates).

ChemGP [@chemgp] supplies the GP-accelerated optimization loop that produces
the trajectories these tools analyze.

# Research impact statement

The library and its predecessor scripts have been used in:

- GP-accelerated Sella saddle point searches [@goswami2026pruning], with
  reproduction package `gpr_sella_repro`
- On-the-fly GP dimer calculations (`otgpd_repro`)
- NEB with machine-learned force fields (`nebmmf_repro`)
- 2D NEB visualization via RMSD projection [@goswami2026rmsd], with
  reproduction package `nebviz_repro`
- ChemGP [@chemgp], whose figure and benchmark environments depend on
  `rgpycrumbs` and which writes eOn-compatible output for it
- The doctoral dissertation [@goswami2025thesis]

External adoption includes eOn [@eon_zenodo], whose documentation
references `rgpycrumbs` as its companion diagnostic and visualization suite;
the atomistic-cookbook [@atomistic_cookbook] (lab-cosmo), which uses
`rgpycrumbs` in its eOn PET-NEB tutorial; and the metatensor ecosystem article
[@metatensor_ecosystem]. Releases are archived on Zenodo [@rgpycrumbs_zenodo].

# AI usage disclosure

Generative AI (Claude, Anthropic) was used as an agentic coding assistant
throughout development of rgpycrumbs and the companion library chemparseplot.
All modules received some degree of AI-assisted development, including surface
fitting kernels, test suites, CLI scripts, and documentation. AI assistance
ranged from line-level completions to full function implementations. The
mathematical core (kernel definitions, gradient derivations, RMSD projections)
was specified by the human author with AI implementing the numerical code.
Test suites were largely AI-generated from human-specified test cases and
reference values. Agentic workflows (Claude Code CLI) were used for multi-file
refactoring, code generation from specifications, and test writing, and
interactive queries for algorithm design and debugging. All generated material
has been reviewed and edited for clarity, correctness, and completeness.
Mathematical implementations were validated against published formulas and
reference implementations (MATLAB, Julia). Test suites run in CI on every
commit.

The core algorithms derive from `gpr_optim` [@gpr_optim], a C++ port of
Olli-Pekka Koistinen's original MATLAB GPR-dimer code, written by hand with
Satish Kamath and Maxim Masterov. The reproduction packages predate any
AI-assisted development. The package architecture (PEP 723 dispatcher,
org-to-rst documentation pipeline, Sphinx integration) is original design
work.

# Acknowledgements

The GP methods build on Koistinen's MATLAB implementation [@koistinen2017].
The C++ port (`gpr_optim`) was developed with Satish Kamath and Maxim
Masterov. Hannes Jonsson supervised the doctoral work.

# References
