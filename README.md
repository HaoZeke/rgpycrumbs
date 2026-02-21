
# Table of Contents

-   [About](#about)
    -   [Ecosystem Overview](#org723df51)
    -   [CLI Design Philosophy](#cli-how)
-   [Usage](#usage)
    -   [Library API](#library-api)
    -   [CLI Tools](#cli-tools)
        -   [eOn](#cli-eon)
-   [Contributing](#contributing)
    -   [Development](#development)
        -   [When is pixi needed?](#org1e75565)
        -   [Versioning](#org4873ddd)
    -   [Release Process](#release-notes)
-   [License](#license)



<a id="about"></a>

# About

![img](https://raw.githubusercontent.com/HaoZeke/rgpycrumbs/refs/heads/main/branding/logo/pycrumbs_logo.webp)

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

A **pure-python** computational library and CLI toolkit for chemical physics
research. `rgpycrumbs` provides both importable library modules for
computational tasks (surface fitting, structure analysis, interpolation) and a
dispatcher-based CLI for running self-contained research scripts.

The library side offers:

-   **Surface fitting** (`rgpycrumbs.surfaces`) &#x2013; JAX-based kernel methods (TPS, RBF, Matern, SE, IMQ) with gradient-enhanced variants for energy landscape interpolation
-   **Structure analysis** (`rgpycrumbs.geom.analysis`) &#x2013; distance matrices, bond matrices, and fragment detection via ASE
-   **IRA matching** (`rgpycrumbs.geom.ira`) &#x2013; iterative rotations and assignments for RMSD-based structure comparison
-   **Interpolation** (`rgpycrumbs.interpolation`) &#x2013; spline interpolation utilities
-   **Data types** (`rgpycrumbs.basetypes`) &#x2013; shared data structures for NEB paths, saddle searches, and molecular geometries

The CLI tools rely on optional dependencies fetched on-demand via PEP 723 + `uv`.


<a id="org723df51"></a>

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
    `uv` only when a script that needs them is executed, keeping the base
    installation lightweight.

-   **Modular & Extensible Tooling:** Each utility is an independent script. This
    modularity simplifies development, testing, and maintenance, as changes to one
    tool cannot inadvertently affect another. New tools can be added to the
    collection without modifying the core dispatcher logic, making the system
    easily extensible.


<a id="usage"></a>

# Usage


<a id="library-api"></a>

## Library API

The library modules can be imported directly:

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

    This script visualizes the energy profile of Nudged Elastic Band (NEB) calculations over optimization steps.
    
    To see the help text for this specific script:
    
        $ python -m rgpycrumbs eon plt-neb --help
        --> Dispatching to: uv run /path/to/rgpycrumbs/eon/plt_neb.py --help
        Usage: plt_neb.py [OPTIONS]
        
          Plots a series of NEB energy paths from .dat files.
        ...
        Options:
          --input-pattern TEXT      Glob pattern for input data files.
          -o, --output-file PATH    Output file name.
          --start INTEGER           Starting file index to plot (inclusive).
          --end INTEGER             Ending file index to plot (exclusive).
          --help                    Show this message and exit.
    
    To plot a specific range of `neb_*.dat` files and save the output:
    
        python -m rgpycrumbs eon plt-neb --start 100 --end 150 -o final_path.pdf
    
    To show the plot interactively without saving:
    
        python -m rgpycrumbs eon plt-neb --start 280

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


<a id="org1e75565"></a>

### When is pixi needed?

[Pixi](https://prefix.dev/) is only needed for features that require **conda-only** packages (not
available on PyPI):

-   `fragments` tests: need `tblite`, `ira`, `pyvista` (conda)
-   `surfaces` tests: may prefer conda `jax` builds

For everything else, `uv` is sufficient.


<a id="org4873ddd"></a>

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

-   The Zenodo DOI for general use.
-   The `wailord` paper for ORCA usage

