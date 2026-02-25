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
date: 23 February 2026
bibliography: paper.bib
---

# Summary

Saddle point searches and minimum energy path calculations are the
rate-limiting step in computational studies of chemical kinetics. Once
converged, the raw output -- positions, energies, forces, eigenvalues at each
image -- requires substantial post-processing before it can be interpreted:
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

1. **What is the energy profile along the path?** The standard representation
   plots energy $E$ against a reaction coordinate $s$ (cumulative path length
   or RMSD). Hermite interpolation using the parallel force component
   $f_\parallel^{(i)} = \mathbf{F}_i \cdot \hat{\boldsymbol{\tau}}_i$ as
   derivative data [@henkelman2000improved] produces physically consistent
   profiles; unconstrained splines through energy values alone oscillate.

2. **What does the energy landscape look like around the path?** A 2D
   representation in coordinates $(r, p)$ -- the RMSD from reactant and product
   structures -- requires aligning each image to the endpoints (handling
   permutational isomers via IRA [@ira] when needed), computing synthetic
   gradients by projecting $f_\parallel$ onto the $(r, p)$ tangent direction,
   and fitting the sparse data with kernel interpolation. Gradient-enhanced
   kernels produce smoother surfaces from fewer points than standard RBF
   methods.

3. **What chemical transformation occurs at the saddle point?** Bond breaking
   and formation events are identified by computing Wiberg Bond Orders (WBO)
   from a GFN2-xTB [@gfn2xtb] single-point calculation at the transition state
   and comparing the bond order matrix to the reactant. Geometric fragment
   detection (scaled covalent radii) provides a fast alternative when electronic
   structure is not available.

These operations recur across saddle point search codes (eOn, ORCA, ASE,
ChemGP) and file formats (.dat/.con, extxyz, HDF5), but existing
implementations are one-off scripts coupled to specific workflows.
`rgpycrumbs` consolidates the computation into tested, importable modules; the
companion library `chemparseplot` [@chemparseplot] handles format-specific
parsing and delegates heavy computation to `rgpycrumbs`.

The algorithms originate from a C++ implementation (`gpr_optim` [@gpr_optim]),
a port of the MATLAB code by Koistinen et al. [@koistinen2017; @koistinen2019],
developed during the doctoral work [@goswami_thesis]. The Python library makes
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
transition state -- the information a chemist needs beyond the barrier height;
(iii) IRA (Iterative Rotations and Assignments) alignment for RMSD calculation
between structures that may be permutational isomers, which is required for
meaningful $(r, p)$ coordinates when the reaction involves atomic exchange.

**Interpolation** (`rgpycrumbs.interpolation`). Hermite spline interpolation
using both energy and force data for 1D profiles along reaction coordinates.

**Data types** (`rgpycrumbs.basetypes`). Shared structures for NEB paths,
saddle point measures, and molecular geometries.

**PLUMED integration** (`rgpycrumbs.plumed`). Direct summation reconstruction
of 2D free energy surfaces from PLUMED metadynamics hills files.

**Test potentials** (`rgpycrumbs.func`). Muller-Brown and other analytical
surfaces for algorithm validation.

## PEP 723 dispatcher

Research workflows in computational chemistry require tools with mutually
exclusive binary dependencies. OVITO (crystal defect analysis) and tblite
(tight-binding electronic structure) cannot coexist in a single Python
environment; JAX and PyTorch builds may also conflict. The `rgpycrumbs.cli`
dispatcher invokes each script in an isolated subprocess via `uv`, using PEP
723 inline metadata to declare per-script dependencies. The fragment detection
script, for instance, declares `tblite`, `pyvista`, and `ase` as its
dependencies and runs in a fresh environment without polluting the host.

![Data flow from raw trajectory data through parsing (`chemparseplot`) and
computation (`rgpycrumbs`) to visualization.](figures/dataflow.pdf){width="100%"}

## Companion libraries

`chemparseplot` [@chemparseplot] handles file parsing (eOn `.dat`/`.con`, ORCA
output, ChemGP HDF5, ASE trajectory formats) and plotting, delegating
computation to `rgpycrumbs`. `pychum` generates input files for ORCA and
NWChem. The three packages form a pipeline from input generation through
computation to visualization.

# State of the field

ASE [@ase] provides atomic simulation infrastructure and NEB implementations
but not 2D landscape interpolation, gradient-enhanced kernel fitting, or
bond-order-based fragment detection at transition states. pymatgen [@pymatgen]
targets materials science database workflows with different scope. catlearn
[@catlearn] implements GP-accelerated NEB within ASE but focuses on the
optimization loop, not post-processing. sGDML [@sgdml] fits kernel-based
molecular force fields for dynamics rather than reaction path analysis. The
`tblite` package [@gfn2xtb] provides the electronic structure backend for
Wiberg Bond Order computation but does not itself perform fragment detection or
transition state analysis.

`rgpycrumbs` occupies the space between a converged calculation and its
chemical interpretation: projecting paths into readable coordinates, fitting
landscapes from sparse data using gradient information, and determining which
bonds break at the saddle point.

# Research impact

The library and its predecessor scripts have been used in:

- GP-accelerated Sella saddle point searches [@gpr_sella], with reproduction
  package `gpr_sella_repro`
- On-the-fly GP dimer calculations (`otgpd_repro`)
- NEB with machine-learned force fields (`nebmmf_repro`)
- 2D NEB visualization methods (`nebviz_repro`)
- The doctoral dissertation [@goswami_thesis]

External adoption includes the atomistic-cookbook [@atomistic_cookbook]
(lab-cosmo), which uses `rgpycrumbs` in its eOn PET-NEB tutorial, and the
metatensor ecosystem article [@metatensor_ecosystem].

# Acknowledgements

The GP methods build on Koistinen's MATLAB implementation [@koistinen2017].
The C++ port (`gpr_optim`) was developed with Satish Kamath and Maxim
Masterov. Hannes Jonsson supervised the doctoral work.

# References
