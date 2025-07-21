
# Table of Contents

-   [About](#org78edd98)
    -   [CLI Design Philosophy](#orgef213ce)
-   [Usage](#orgbdc64ce)
    -   [CLI Tools](#org86529f6)
        -   [EON](#org5ab1cb7)
-   [Contributing](#orgac70168)
-   [License](#org7a5cb11)



<a id="org78edd98"></a>

# About

![img](branding/logo/pycrumbs_logo.webp)

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

A **pure-python** module of helpful scripts / CLIs I have written mostly for quick
and dirty parsing / plotting of optimization methods. Loosely grouped. Not
typically optimal at all.

Most of these rely heavily on optional dependencies, with the exception of the
`_aux` helpers (pure Python).


<a id="orgef213ce"></a>

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
    package is minimal, with its only dependency being the `click` library for the
    CLI dispatcher. Heavy scientific libraries like `matplotlib`  are not part of
    the base installation. They are fetched by `uv` only when a script that needs
    them is executed, ensuring the user's base Python environment remains clean
    and lightweight.

-   **Modular & Extensible Tooling:** Each utility is an independent script. This
    modularity simplifies development, testing, and maintenance, as changes to one
    tool cannot inadvertently affect another. New tools can be added to the
    collection without modifying the core dispatcher logic, making the system
    easily extensible.


<a id="orgbdc64ce"></a>

# Usage


<a id="org86529f6"></a>

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


<a id="org5ab1cb7"></a>

### EON

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


<a id="orgac70168"></a>

# Contributing

All contributions are welcome, but for the CLI tools please follow [established
best practices](https://realpython.com/python-script-structure/).


<a id="org7a5cb11"></a>

# License

MIT. However, this is an academic resource, so **please cite** as much as possible
via:

-   The Zenodo DOI for general use.
-   The `wailord` paper for ORCA usage

