
# Table of Contents

-   [About](#about)
    -   [Ecosystem Overview](#ecosys)
    -   [CLI Design Philosophy](#cli-how)
-   [Usage](#usage)
    -   [Library API](#library-api)
    -   [CLI Tools](#cli-tools)
        -   [eOn](#cli-eon)
-   [Contributing](#contributing)
    -   [Development](#development)
        -   [Branch Structure](#dev:branch)
        -   [When is pixi needed?](#dev:whypixi)
        -   [Versioning](#dev:versions)
    -   [Release Process](#release-notes)
-   [License](#license)



<a id="about"></a>

# About

![img](https://raw.githubusercontent.com/HaoZeke/rgpycrumbs/refs/heads/main/branding/logo/pycrumbs_logo.webp)

[![Tests](https://github.com/HaoZeke/rgpycrumbs/actions/workflows/ci_test.yml/badge.svg)](https://github.com/HaoZeke/rgpycrumbs/actions/workflows/ci_test.yml)
[![Linting](https://github.com/HaoZeke/rgpycrumbs/actions/workflows/ci_prek.yml/badge.svg)](https://github.com/HaoZeke/rgpycrumbs/actions/workflows/ci_prek.yml)
[![Docs](https://github.com/HaoZeke/rgpycrumbs/actions/workflows/ci_docs.yml/badge.svg)](https://github.com/HaoZeke/rgpycrumbs/actions/workflows/ci_docs.yml)
[![PyPI](https://img.shields.io/pypi/v/rgpycrumbs)](https://pypi.org/project/rgpycrumbs/)
[![Python](https://img.shields.io/pypi/pyversions/rgpycrumbs)](https://pypi.org/project/rgpycrumbs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![One Good Tutorial docs checklist v1: adopted](https://onegoodtutorial.org/badge/adopted-v1.svg)](https://onegoodtutorial.org/about/badge/?v=1)
[![Benchmarks](https://img.shields.io/badge/benchmarks-asv--perch-orange)](https://github.com/HaoZeke/asv-perch)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![DOI](https://zenodo.org/badge/795596895.svg)](https://doi.org/10.5281/zenodo.18529798)

A **pure-python** computational library and CLI toolkit for chemical physics
research. `rgpycrumbs` provides both importable library modules for
computational tasks (surface fitting, structure analysis, interpolation) and a
dispatcher-based CLI for running self-contained research scripts.

Heavy optional dependencies (JAX, SciPy, ASE) are resolved lazily at first use.
A bare `pip install rgpycrumbs` gives the full API surface; the actual backend
libraries load on demand from the current environment, a shared cache, or (with
`RGPYCRUMBS_AUTO_DEPS=1`) via automatic `uv` installation. CUDA-aware
resolution avoids pulling GPU libraries on CPU-only machines.

The library side offers:

-   **Surface fitting** (`rgpycrumbs.surfaces`) &#x2013; JAX-based kernel methods (TPS, RBF, Matern, SE, IMQ) with gradient-enhanced variants for energy landscape interpolation
-   **Structure analysis** (`rgpycrumbs.geom.analysis`) &#x2013; distance matrices, bond matrices, and fragment detection via ASE
-   **IRA matching** (`rgpycrumbs.geom.ira`) &#x2013; iterative rotations and assignments for RMSD-based structure comparison
-   **Interpolation** (`rgpycrumbs.interpolation`) &#x2013; spline interpolation utilities
-   **Data types** (`rgpycrumbs.basetypes`) &#x2013; shared data structures for NEB paths, saddle searches, and molecular geometries

The CLI tools rely on optional dependencies fetched on-demand via PEP 723 + `uv`.


<a id="ecosys"></a>

## Ecosystem Overview

`rgpycrumbs` is the central hub of an interlinked suite of libraries.

![img](branding/logo/ecosystem.png)


<a id="cli-how"></a>

## CLI Design Philosophy

The library is designed with the following principles in mind:

-   **Dispatcher-Based Architecture:** The top-level `rgpycrumbs.cli` command acts as a
    lightweight dispatcher. It does not contain the core logic of the tools
    itself. Instead, it parses user commands to identify the target script and
    then invokes it in an isolated subprocess using the `uv` runner. This provides
    a unified command-line interface while keeping the tools decoupled.

-   **Isolated & Reproducible Execution:** Each script is a self-contained unit that
    declares its own dependencies via [PEP 723](https://peps.python.org/pep-0723/) metadata. The `uv` runner uses this
    information to resolve and install the exact required packages into a
    temporary, cached environment on-demand. This design guarantees
    reproducibility and completely eliminates the risk of dependency conflicts
    between different tools in the collection.

-   **Lightweight Core, On-Demand Dependencies:** The installable `rgpycrumbs`
    package has minimal core dependencies (`click`, `numpy`). Heavy scientific
    libraries are available as optional extras (e.g. `pip install
      rgpycrumbs[surfaces]` for JAX). For CLI tools, dependencies are fetched by
    `uv` only when a script that needs them is executed. For library modules,
    `ensure_import` resolves dependencies at first use when `RGPYCRUMBS_AUTO_DEPS=1`
    is set, with CUDA-aware resolution that avoids pulling GPU libraries on
    CPU-only machines. The base installation stays lightweight either way.

-   **Modular & Extensible Tooling:** Each utility is an independent script. This
    modularity simplifies development, testing, and maintenance, as changes to one
    tool cannot inadvertently affect another. New tools can be added to the
    collection without modifying the core dispatcher logic, making the system
    easily extensible.


<a id="usage"></a>

# Usage


<a id="library-api"></a>

## Library API

The library modules can be imported directly. Dependencies resolve
automatically when `RGPYCRUMBS_AUTO_DEPS=1` is set (requires `uv` on PATH),
or install extras explicitly:

    # Surface fitting (requires jax: pip install rgpycrumbs[surfaces])
    from rgpycrumbs.surfaces import get_surface_model
    model = get_surface_model("tps")
    
    # Structure analysis (requires ase, scipy: pip install rgpycrumbs[analysis])
    from rgpycrumbs.geom.analysis import analyze_structure
    
    # Spline interpolation (requires scipy: pip install rgpycrumbs[interpolation])
    from rgpycrumbs.interpolation import spline_interp
    
    # Data types (no extra deps)
    from rgpycrumbs.basetypes import nebpath, SaddleMeasure


<a id="cli-tools"></a>

## CLI Tools

The general command structure is:

    python -m rgpycrumbs.cli [subcommand-group] [script-name] [script-options]

You can see the list of available command groups:

    $ python -m rgpycrumbs.cli --help
    Usage: rgpycrumbs [OPTIONS] COMMAND [ARGS]...
    
      A dispatcher that runs self-contained scripts using 'uv'.
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      eon  Dispatches to a script within the 'eon' submodule.


<a id="cli-eon"></a>

### eOn

-   Plotting NEB Paths (`plt-neb`)

    This script visualizes the energy landscape of Nudged Elastic Band (NEB) calculations,
    generating 2D surface plots with optional structure rendering.
    
    The default `grad_imq` method uses gradient-enhanced Inverse Multiquadric interpolation
    on 2D RMSD projections [1]. The approach projects high-dimensional structures onto
    2D coordinates (reactant distance `r` vs product distance `p`) and fits a smooth
    surface using energy values and their gradients.
    
    [1] R. Goswami, "Two-dimensional RMSD projections for reaction path visualization
    and validation," *MethodsX*, p. 103851, Mar. 2026,
    doi: [10.1016/j.mex.2026.103851](https://doi.org/10.1016/j.mex.2026.103851).
    generating 2D surface plots with optional structure rendering.
    
    -   Basic Usage
    
            python -m rgpycrumbs eon plt-neb --con-file trajectory.con --plot-type landscape -o neb_landscape.png
    
    -   Key Options
    
        <table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
        
        
        <colgroup>
        <col  class="org-left" />
        
        <col  class="org-left" />
        
        <col  class="org-left" />
        </colgroup>
        <thead>
        <tr>
        <th scope="col" class="org-left">Option</th>
        <th scope="col" class="org-left">Description</th>
        <th scope="col" class="org-left">Default</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td class="org-left"><code>--con-file PATH</code></td>
        <td class="org-left">Trajectory file with NEB images</td>
        <td class="org-left">None</td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--plot-type</code></td>
        <td class="org-left"><code>landscape</code> (2D surface) or <code>profile</code></td>
        <td class="org-left"><code>profile</code></td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--surface-type</code></td>
        <td class="org-left">Surface method: <code>grad_imq</code>, <code>grad_matern</code>, <code>grad_imq_ny</code>, <code>rbf</code></td>
        <td class="org-left"><code>grad_imq</code></td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--project-path</code> / <code>--no-project-path</code></td>
        <td class="org-left">Project to reaction valley coordinates</td>
        <td class="org-left"><code>--project-path</code></td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--plot-structures</code></td>
        <td class="org-left">Structure strip: <code>none</code>, <code>all</code>, <code>crit_points</code></td>
        <td class="org-left"><code>none</code></td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--show-legend</code></td>
        <td class="org-left">Show colorbar legend</td>
        <td class="org-left">Off</td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--show-pts</code> / <code>--no-show-pts</code></td>
        <td class="org-left">Show data points on surface</td>
        <td class="org-left"><code>--show-pts</code></td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--landscape-path</code></td>
        <td class="org-left">Path overlay: <code>final</code>, <code>all</code>, <code>none</code></td>
        <td class="org-left"><code>final</code></td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--ira-kmax</code></td>
        <td class="org-left">kmax factor for IRA RMSD calculation</td>
        <td class="org-left">1.8</td>
        </tr>
        
        <tr>
        <td class="org-left"><code>-o PATH</code></td>
        <td class="org-left">Output image filename</td>
        <td class="org-left">None (display)</td>
        </tr>
        </tbody>
        </table>
        
        Use `--landscape-path all` to overlay all optimization steps and visualize convergence.
        This shows the full trajectory from initial guess to final path [1].
        
        <table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
        
        
        <colgroup>
        <col  class="org-left" />
        
        <col  class="org-left" />
        
        <col  class="org-left" />
        </colgroup>
        <tbody>
        <tr>
        <td class="org-left"><code>--show-pts</code> / <code>--no-show-pts</code></td>
        <td class="org-left">Show data points on surface</td>
        <td class="org-left"><code>--show-pts</code></td>
        </tr>
        
        <tr>
        <td class="org-left"><code>--ira-kmax</code></td>
        <td class="org-left">kmax factor for IRA RMSD calculation</td>
        <td class="org-left">1.8</td>
        </tr>
        
        <tr>
        <td class="org-left"><code>-o PATH</code></td>
        <td class="org-left">Output image filename</td>
        <td class="org-left">None (display)</td>
        </tr>
        </tbody>
        </table>
    
    -   Examples
    
        Full landscape with gradient-enhanced IMQ surface and critical point structures:
        
            python -m rgpycrumbs eon plt-neb \
              --con-file neb.con \
              --plot-type landscape \
              --project-path \
              --plot-structures crit_points \
              --surface-type grad_imq \
              --ira-kmax 14 \
              --show-legend \
              -o neb_landscape.png
        
        Surface-only plot without structure strip:
        
            python -m rgpycrumbs eon plt-neb \
              --plot-type landscape \
              --surface-type grad_matern \
              --no-show-pts \
              -o surface.png
        
        Convergence visualization with all optimization steps:
        
            python -m rgpycrumbs eon plt-neb \
              --con-file neb.con \
              --plot-type landscape \
              --landscape-path all \
              --surface-type grad_imq \
              --show-legend \
              -o neb_convergence.png
    
    -   Surface Methods
    
        -   **`grad_imq`:** Gradient-enhanced Inverse Multiquadric (recommended, uses energy + gradients)
        -   **`grad_matern`:** Gradient-enhanced Matérn 5/2 (uses energy + gradients)
        -   **`grad_imq_ny`:** Nystrom-approximated grad<sub>imq</sub> for large datasets (>1000 points)
        -   **`rbf`:** Radial Basis Function / Thin Plate Spline (fast, no gradients)

-   Splitting CON files (`con-splitter`)

    This script takes a multi-image trajectory file (e.g., from a finished NEB
    calculation) and splits it into individual frame files, creating an input file
    for a new calculation.
    
    To split a trajectory file:
    
        rgpycrumbs eon con-splitter neb_final_path.con -o initial_images
    
    This will create a directory named `initial_images` containing `ipath_000.con`,
    `ipath_001.con`, etc., along with an `ipath.dat` file listing their paths.


<a id="contributing"></a>

# Contributing

All contributions are welcome, but for the CLI tools please follow [established
best practices](https://realpython.com/python-script-structure/).


<a id="development"></a>

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) as the primary development tool with
[`hatchling`](https://hatch.pypa.io/) + [`hatch-vcs`](https://github.com/ofek/hatch-vcs) for building and versioning.

    # Clone and install in development mode with test dependencies
    uv sync --extra test
    
    # Run the pure tests (no heavy optional deps)
    uv run pytest -m pure
    
    # Run interpolation tests (needs scipy)
    uv run --extra interpolation pytest -m interpolation


<a id="dev:branch"></a>

### Branch Structure

Development happens on the `main` branch. The `readme` branch is an
auto-generated orphan containing only the rendered `README.md` and branding
assets; it is the GitHub default branch.


<a id="dev:whypixi"></a>

### When is pixi needed?

[Pixi](https://prefix.dev/) is only needed for features that require **conda-only** packages (not
available on PyPI):

-   `fragments` tests: need `tblite`, `ira`, `pyvista` (conda)
-   `surfaces` tests: may prefer conda `jax` builds

For everything else, `uv` is sufficient.


<a id="dev:versions"></a>

### Versioning

Versions are derived automatically from **git tags** via `hatch-vcs`
(setuptools-scm). There is no manual version field; the version is the latest
tag (e.g. `v1.0.0` → `1.0.0`). Between tags, dev versions are generated
automatically (e.g. `1.0.1.dev3+gabcdef`).


<a id="release-notes"></a>

## Release Process

    # 1. Ensure tests pass
    uv run --extra test pytest -m pure
    
    # 2. Build changelog (uses towncrier fragments in docs/newsfragments/)
    uvx towncrier build --version "v1.0.0"
    
    # 3. Commit the changelog
    git add CHANGELOG.rst && git commit -m "doc: release notes for v1.0.0"
    
    # 4. Tag the release (hatch-vcs derives the version from this tag)
    git tag -a v1.0.0 -m "Version 1.0.0"
    
    # 5. Build and publish
    uv build
    uvx twine upload dist/*


<a id="license"></a>

# License

MIT. However, this is an academic resource, so **please cite** as much as possible
via:

-   The [Zenodo DOI](https://doi.org/10.5281/zenodo.18529798) for general use.
-   The `wailord` paper for ORCA usage


## Optional Dependencies

rgpycrumbs uses lazy imports for optional dependencies. Install features as needed:

```bash
# Surface fitting (JAX)
pip install "rgpycrumbs[surfaces]"

# Interpolation (scipy)
pip install "rgpycrumbs[interpolation]"

# All optional dependencies
pip install "rgpycrumbs[all]"
```

### Automatic Installation

Enable on-demand installation with environment variable:

```bash
export RGPYCRUMBS_AUTO_DEPS=1
```

When enabled, missing optional dependencies are automatically installed on first use.
See [[file:docs/orgmode/howto/install.org][Installation Guide]] for details.
