---
name: rgpycrumbs-add-tool
description: Add a command-line tool to rgpycrumbs. Covers where the script goes, how its file name becomes the command, the PEP 723 header the dispatcher installs from, keeping library modules out of the command list, tests with the pure and pep723 markers, the changelog fragment, and the docs pages to update. Use when asked to add or rename an rgpycrumbs subcommand, or when a new script is missing from `rgpycrumbs GROUP --help`.
license: MIT
compatibility: An rgpycrumbs checkout with Python 3.11 or newer and uv (uv sync --extra test). The check script needs the core install only and runs offline.
---

# Adding an rgpycrumbs tool

## Where it goes and what it is called

The dispatcher builds its command list from the files, so there is no registry to edit.
`rgpycrumbs/GROUP/NAME.py` becomes `rgpycrumbs GROUP NAME` with underscores turned into dashes: `rgpycrumbs/geom/count_atoms.py` is `rgpycrumbs geom count-atoms`.
A new directory under `rgpycrumbs/` becomes a group once it holds a command; give it an `__init__.py` so that tests can import the module.

Every other `.py` file directly in a group directory is listed as a command too, although running a library module that way does no work: it exits 0 without output, or stops with `ModuleNotFoundError` when uv runs it without the packages it imports.
Name helpers `_name.py`, or put them in a subpackage such as `rgpycrumbs/geom/api/`.
Only `hdf5_io`, `plotting`, `utils`, `helpers` and `seed_dimers` are left out by name.

## The script

[assets/count_atoms.py](assets/count_atoms.py) is a complete tool:

- A PEP 723 block naming every third-party import, with `requires-python = ">=3.11"` like the existing tools.
  The dispatcher installs exactly this list under uv, so a missing entry is an `ImportError` for users even when the tests pass in a full environment.
- The work in a function the tests can import (`count_elements`), and a thin click command around it, as `docs/orgmode/dev/modular_design.org` describes.
- A module docstring; existing tools add `.. versionadded::` with the release that first ships them.

Try it without uv while writing it:

```bash
rgpycrumbs --dev geom count-atoms molecule.xyz
```

## Tests

- `tests/test_NAME.py`, calling the functions directly and the command through `click.testing.CliRunner`.
  Mark the tests that need only the `test` extra with `pytestmark = pytest.mark.pure`; CI runs `uv run --extra test pytest -m pure` on Python 3.10, 3.11 and 3.12.
  The other markers (`interpolation`, `align`, `surfaces`, `fragments`, `eon`, `ptm`, `ira`) select tests that need extra packages; they are declared in `pyproject.toml`.
- The header check, which fails when an import is not covered by the PEP 723 dependencies:

  ```bash
  uv run --extra test pytest --pep723-check -m pep723 \
    --override-ini="python_files=" --ignore=rgpycrumbs/chemgp rgpycrumbs/
  ```

## Changelog and docs

- A towncrier fragment, `docs/newsfragments/NAME.added.rst`, one paragraph in reStructuredText (double backticks for code).
  The types are `security`, `removed`, `deprecated`, `added`, `dev`, `changed`, `fixed` and `misc`.
  `.gitignore` has `*.rst`, so add the fragment with `git add -f`.
- A page `docs/orgmode/tools/GROUP/NAME.org`, added to the toctree in the `#+BEGIN_EXPORT rst` block of `docs/orgmode/tools/GROUP/index.org`, and a line in the site map `docs/source/llms.txt`.
- `python scripts/llms/build_llms.py` to refresh `llms-full.txt`; CI fails while it is stale or while `docs/source/llms.txt` links a page that does not exist.

## Check

```bash
python scripts/scaffold_check.py
```

copies the installed package into a temporary directory, adds a `demo` group holding [assets/count_atoms.py](assets/count_atoms.py), a `_private.py`, a `plotting.py` and a header-less `analysis.py`, and checks that `rgpycrumbs demo --help` lists `count-atoms` and `analysis` but not the other two, that the template's PEP 723 block parses, and that `rgpycrumbs --dev demo count-atoms` counts the atoms of a water molecule.
