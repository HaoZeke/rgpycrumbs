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
date: 7 April 2026
bibliography: paper.bib
---


# Summary

Saddle point searches and minimum energy path calculations are the
rate-limiting step in computational studies of chemical kinetics. Once
converged, the raw output &#x2013; positions, energies, forces, eigenvalues at each
image &#x2013; requires substantial post-processing before it can be interpreted:
energy profiles must be interpolated with force consistency, paths must be
projected into low-dimensional coordinate systems, chemical transformations at
the transition state must be identified, and sparse landscape data must be
interpolated onto grids for visualization. `rgpycrumbs` provides the
computational modules for these tasks, together with a PEP 723-based CLI
dispatcher that manages the execution of analysis scripts whose binary
dependencies would otherwise conflict.


# Statement of need

A converged saddle point search produces a transition state geometry and the
minimum energy path connecting it to adjacent minima. Three questions follow
immediately:

1.  **What is the energy profile along the path?** The standard representation
    plots energy $E$ against a reaction coordinate $s$ (cumulative path length
    or RMSD). Hermite interpolation using the parallel force component
    $f_\parallel^{(i)} = \mathbf{F}_i \cdot \hat{\boldsymbol{\tau}}_i$ as
    derivative data produces physically consistent
    profiles; unconstrained splines through energy values alone oscillate.

2.  **What does the energy landscape look like around the path?** A 2D
    representation in coordinates $(r, p)$ &#x2013; the RMSD from reactant and product
    structures &#x2013; requires aligning each image to the endpoints (handling
    permutational isomers via IRA when needed), computing synthetic
    gradients by projecting $f_\parallel$ onto the $(r, p)$ tangent direction,
    and fitting the sparse data with kernel interpolation. Gradient-enhanced
    kernels produce smoother surfaces from fewer points than standard RBF
    methods.

3.  **What chemical transformation occurs at the saddle point?** Bond breaking
    and formation events are identified by computing Wiberg Bond Orders (WBO)
    from a GFN2-xTB single-point calculation at the transition state
    and comparing the bond order matrix to the reactant. Geometric fragment
    detection (scaled covalent radii) provides a fast alternative when electronic
    structure is not available.

These operations recur across saddle point search codes (eOn, ORCA, ASE,
ChemGP) and file formats (.dat/.con, extxyz, HDF5), but existing
implementations are one-off scripts coupled to specific workflows.
`rgpycrumbs` consolidates the computation into tested, importable modules; the
companion library `chemparseplot` handles format-specific
parsing and delegates heavy computation to `rgpycrumbs`.

The algorithms originate from a C++ implementation (`gpr_optim` ),
a port of the MATLAB code by Koistinen et al. ,
developed during the doctoral work . The Python library makes
these methods accessible without C++ compilation.


# Software design

The package separates into library modules and a CLI dispatcher.

![Architecture of `rgpycrumbs`. Library modules (left) provide importable APIs.
The PEP 723 dispatcher (right) manages isolated environments for CLI scripts
with conflicting binary requirements.](figures/architecture.pdf){width="100%"}


## Library modules

**Surface fitting** (`rgpycrumbs.surfaces`). JAX-based Gaussian process
regression with six kernel families (Matern 3/2, inverse multiquadric, rational
quadratic, squared exponential, thin-plate spline, RBF), each available in
standard and gradient-enhanced variants. Nystrom approximation handles larger
datasets. These interpolate 2D energy or eigenvalue landscapes from sparse NEB
data in the $(r, p)$ coordinate system.

**Structural analysis** (`rgpycrumbs.geom`). Three capabilities: (i) distance
and bond matrix construction from ASE `Atoms` objects, with fragment detection
via connected components of the bond graph; (ii) Wiberg Bond Order analysis
through GFN2-xTB (`tblite`), which identifies which bonds break or form at the
transition state &#x2013; the information a chemist needs beyond the barrier height;
(iii) IRA (Iterative Rotations and Assignments) alignment for RMSD calculation
between structures that may be permutational isomers, which is required for
meaningful $(r, p)$ coordinates when the reaction involves atomic exchange.

**Interpolation** (`rgpycrumbs.interpolation`). Hermite spline interpolation
using both energy and force data for 1D profiles along reaction coordinates.

**Data types** (`rgpycrumbs.basetypes`). Shared structures for NEB paths,
saddle point measures, and molecular geometries.

**PLUMED integration** (`rgpycrumbs.plumed`). A CLI for free energy surface
reconstruction from metadynamics simulations. The HILLS file parsing and FES
kernel summation are provided by `chemparseplot.parse.plumed`; `rgpycrumbs`
retains the reconstruction workflow that chains parsing, minima detection, and
visualization into a single script.

**Test potentials** (`rgpycrumbs.func`). Muller-Brown and other analytical
surfaces for algorithm validation.


## On-demand dependency resolution

Research workflows in computational chemistry require tools with mutually
exclusive binary dependencies. OVITO (crystal defect analysis) and tblite
(tight-binding electronic structure) cannot coexist in a single Python
environment; JAX and PyTorch builds may also conflict.

The package addresses this through two complementary mechanisms that share the
same philosophy: a lightweight core with dependencies resolved on demand.

For CLI scripts, the `rgpycrumbs.cli` dispatcher invokes each script in an
isolated subprocess via `uv`, using PEP 723 inline metadata to declare
per-script dependencies. The fragment detection script, for instance, declares
`tblite`, `pyvista`, and `ase` as its dependencies and runs in a fresh
environment without polluting the host.

For library modules, `ensure_import` resolves dependencies at first use through
a priority chain: current environment, parent environment fallback, XDG cache
lookup, and (when opted in via `RGPYCRUMBS_AUTO_DEPS=1`) automatic installation
via `uv pip install --target` into a persistent cache directory. The resolver
detects CUDA availability and selects CPU-only package variants when no GPU is
present, avoiding unnecessary downloads of GPU-specific binary dependencies.
This means `pip install rgpycrumbs` provides the full import surface; heavy
dependencies materialize only when first accessed.

The correctness of the PEP 723 metadata is enforced by `pytest-pep723`
, a pytest plugin developed alongside `rgpycrumbs` that
statically verifies every import in a dispatched script is declared in its
inline metadata block. The plugin parses the `# /// script` dependencies,
extracts all import statements via Python's AST, and reports uncovered imports.
This catches a class of bug where a developer adds a new import but forgets to
update the inline metadata &#x2013; a failure that only manifests at dispatch time in
a clean environment. The plugin is published on PyPI and runs in CI on every
push.

![Data flow from raw trajectory data through parsing (`chemparseplot`) and
computation (`rgpycrumbs`) to visualization.](figures/dataflow.pdf){width="100%"}


## Companion libraries

`chemparseplot` handles file parsing (eOn `.dat` / `.con`, ORCA
output, ChemGP HDF5 and JSONL, PLUMED HILLS, ASE trajectory formats) and
plotting, including publication-quality NEB landscape and energy profile CLIs.
The separation follows a clear boundary: `rgpycrumbs` provides computational
kernels (surface fitting, alignment, interpolation) and `chemparseplot`
provides I/O, visualization, and format-specific logic. `pychum` generates
input files for ORCA and eOn. The three packages form a pipeline from input
generation through computation to visualization, with ChemGP
providing the GP-accelerated optimization loop that produces
the data these tools analyze.


# State of the field

ASE provides atomic simulation infrastructure and NEB implementations
but not 2D landscape interpolation, gradient-enhanced kernel fitting, or
bond-order-based fragment detection at transition states. pymatgen 
targets materials science database workflows with different scope. catlearn
implements GP-accelerated NEB within ASE but focuses on the
optimization loop, not post-processing. sGDML fits kernel-based
molecular force fields for dynamics rather than reaction path analysis. The
`tblite` package provides the electronic structure backend for
Wiberg Bond Order computation but does not itself perform fragment detection or
transition state analysis.

`rgpycrumbs` occupies the space between a converged calculation and its
chemical interpretation: projecting paths into readable coordinates, fitting
landscapes from sparse data using gradient information, and determining which
bonds break at the saddle point.


# Research impact

The library and its predecessor scripts have been used in:

-   GP-accelerated Sella saddle point searches , with reproduction
    package `gpr_sella_repro`
-   On-the-fly GP dimer calculations (`otgpd_repro`)
-   NEB with machine-learned force fields (`nebmmf_repro`)
-   2D NEB visualization via RMSD projection , with
    reproduction package `nebviz_repro`
-   ChemGP , a Rust-based GP-accelerated optimizer for saddle point
    searches and NEB calculations, which uses `rgpycrumbs` kernels for surface
    fitting and `chemparseplot` for figure generation
-   The doctoral dissertation

External adoption includes the atomistic-cookbook 
(lab-cosmo), which uses `rgpycrumbs` in its eOn PET-NEB tutorial, and the
metatensor ecosystem article .


# Acknowledgements

The GP methods build on Koistinen's MATLAB implementation .
The C++ port (`gpr_optim`) was developed with Satish Kamath and Maxim
Masterov. Hannes Jonsson supervised the doctoral work.


# References

