# AGENTS.md

Working notes for changing or scripting against rgpycrumbs.
The manual is Org under `docs/orgmode/`, published at <https://rgpycrumbs.rgoswami.me/>.
`docs/source/llms.txt` is its hand-written map, served as <https://rgpycrumbs.rgoswami.me/llms.txt>.
`llms-full.txt` at the repository root has every page but the API reference in one file.
Task guides with scripts that CI runs are under `skills/`.

## What rgpycrumbs is

- A command dispatcher.
  `rgpycrumbs GROUP TOOL ARGS` runs the script `rgpycrumbs/GROUP/TOOL.py`, with `_` for each `-` in `TOOL`.
  By default it runs the script under `uv run --script`, so the tool installs what its Python Enhancement Proposal (PEP) 723 header names before it starts.
  Pitfalls says when the current Python runs it instead.
  The groups are `chemgp`, `eon`, `func`, `geom`, `orca`, `parsers`, `plumed`, `prefix`, `run`, `surfaces`, `surrogate` and `uma`.
  `rgpycrumbs config show` prints the merged configuration.
- A library.
  `docs/orgmode/explanation/public_api.org` lists what is public, and most of it is in `rgpycrumbs.api`.
  `import rgpycrumbs` needs only click, numpy and rich.
  `rgpycrumbs.surfaces`, `.interpolation` and `.geom` load on first use and raise `ImportError` when their packages (jax, scipy, ase) are missing.

## Layout

- `rgpycrumbs/cli.py`: the dispatcher, with command discovery, pins and `uv run`.
- `rgpycrumbs/config.py`: the layers of `rgpkgs.toml`.
- `rgpycrumbs/locks.py`: the readers of lock files and of software bill of materials (SBOM) files.
- `rgpycrumbs/_aux.py`: `ensure_import`, which installs packages at run time into `~/.cache/rgpycrumbs/deps`.
- `rgpycrumbs/GROUP/`: one directory per command group, with its tools and library modules.
- `tests/`: the pytest suite, split by markers (see Test).
- `docs/orgmode/`: the manual sources in Org, with the Sphinx side in `docs/source/`.
- `docs/orgmode/tutorials/data/`: sample eOn runs that the tutorials and skills read.
- `docs/newsfragments/`: towncrier fragments for the next release, created on demand.
- `skills/`: task guides, each a `SKILL.md` with its `scripts/`.
- `scripts/`: the Org doctest runner, the README export, `llms/` and `skills/`.
- `readme_src.org`: the source of `README.md`.

## Install

```bash
uv sync --extra test          # Python 3.10 or newer; CI uses --all-extras
uv run rgpycrumbs --help
```

Dispatched tools need Python 3.11 or newer and `uv` on `PATH`.
For the suites that need conda packages (ira, tblite, ovito, eOn), use pixi.
Its environments are `test`, `surfaces`, `fragments`, `eonmlflow`, `ptm`, `docs` and `bench`, for `linux-64` only.

## Test

These are the CI jobs in `.github/workflows/ci_test.yml`, `ci_doctest.yml` and `ci_tutorials.yml`:

```bash
uv run --extra test pytest -m pure
uv run --extra test pytest --pep723-check -m pep723 \
  --override-ini="python_files=" --ignore=rgpycrumbs/chemgp rgpycrumbs/
uv run --extra test --with "scipy>=1.11" pytest -m interpolation
uv run --extra test --with "ase>=3.22" --with "scipy>=1.11" pytest -m align
pixi r -e fragments pytest -m fragments
pixi r -e surfaces pytest -m surfaces
uv run --extra test python3 scripts/run_org_doctests.py docs/orgmode/tutorials/*.org
```

The markers are declared in `pyproject.toml`: `pure`, `interpolation`, `align`, `surfaces`, `fragments`, `eon`, `ptm`, `ira` and `slow`.
A module whose packages are missing skips itself (`skip_if_not_env` in `tests/conftest.py`), so add `-rs` to see what did not run.
A tool that imports a package its header does not list fails the PEP 723 check.

and in `.github/workflows/ci_skills.yml`:

```bash
python scripts/llms/build_llms.py --check   # llms-full.txt is current
python -m unittest discover -s scripts/llms -p "test_*.py"   # the generator
python scripts/skills/run_skills.py --no-run   # every SKILL.md is well formed
uv run python scripts/skills/run_skills.py --only rgpycrumbs-cli --only rgpycrumbs-add-tool
```

The first three need Python 3.11 or newer and nothing else.
The nudged elastic band (NEB) skill needs the plot stack in the running environment.
The workflow shows the `uv run --with ...` line.

## Lint

```bash
uvx prek run -a
```

runs ruff (line length 90, Python 3.10 target), ruff-format, taplo, codespell and the `check-yaml` and `check-toml` hooks from `prek.toml`.
When `readme_src.org` changes and Emacs is installed, it also regenerates `README.md`.

## Docs

The docs build is one pixi task.

```bash
pixi r docbld
```

It runs `genfigs` in the `test` environment, then exports the Org files to `docs/source/*.rst` with `emacs --batch --load export.el`, run from `docs/`.
On first use the export fetches `ox-rst` from Milkypostman's Emacs Lisp Package Archive (MELPA).
Last, `sphinx-build docs/source docs/build` builds the site.
The `.rst` files are generated and ignored by git, so edit the Org files.

## Common tasks

- Run a tool, find out what it installs, or pin versions: `skills/rgpycrumbs-cli/`.
- Barriers and figures from an eOn NEB run: `skills/rgpycrumbs-neb-plots/`.
- Add a tool: `skills/rgpycrumbs-add-tool/`.
- Edit the manual: change the `.org` file.
  A new page also goes in a toctree (inside a `#+BEGIN_EXPORT rst` block) and in `docs/source/llms.txt`.
  Then run `python scripts/llms/build_llms.py` (settings in `scripts/llms/llms.toml`) to refresh `llms-full.txt`.
- Edit the README: change `readme_src.org`, then run `bash scripts/org_to_md.sh readme_src.org README.md` with Emacs 30.2 (Org 9.7.11).
  Emacs 29.3 (Org 9.6.15) writes 13 more blank lines than the committed file.
- Record a change: a fragment `docs/newsfragments/NAME.TYPE.rst`, one paragraph of reStructuredText, where `TYPE` is `security`, `removed`, `deprecated`, `added`, `dev`, `changed`, `fixed` or `misc`.
  Prefix `NAME` with `+` when there is no issue number.
  Add it with `git add -f`, since `.gitignore` has `*.rst`.
  `towncrier build` writes `CHANGELOG.rst` at release time, so do not edit it.
- Commit subjects are conventional (`type(scope): subject`) with cocogitto's default types.
  `cog.toml` adds none, so write `docs`, not `doc`.
  `docs/orgmode/contributing/release.org` covers releases.

## Pitfalls

- The version comes from the git tag through hatch-vcs.
  Building or installing the package writes it to `rgpycrumbs/_version.py`, and `rgpycrumbs.__version__` reads that file.
  A source tree that was never built or installed has no such file, so there `rgpycrumbs.__version__` is `unknown`.
  Nothing in the tree sets the version, and `scripts/check_version.py` fails when `pixi.toml` sets one.
- `rgpycrumbs GROUP TOOL --help` passes `--help` to the script, so uv installs the tool's whole dependency list first.
  The pages under <https://rgpycrumbs.rgoswami.me/tools/> list the options without installing anything.
- Many tool headers take readcon and chemparseplot from git through `[tool.uv.sources]`.
  Building readcon from git needs Rust 1.85 or newer.
  With an older Cargo, uv stops with "feature `edition2024` is required".
  Installing the PyPI releases into the running environment avoids the build (`skills/rgpycrumbs-neb-plots/`).
- When uv is missing, the dispatcher runs a tool with the current Python instead.
  It does the same when readcon, matplotlib, polars, ase and chemparseplot all import.
  The tool then uses the versions installed there, not those its header asks for.
  `RGPKGS_FORCE_UV=1` forces uv.
  `RGPYCRUMBS_DEV=1` turns on `--dev`, but `RGPKGS_DEV=1` does not.
- Sixteen listed commands, such as `geom analysis` and `func muller-brown`, are library modules and do no work when run.
  Under uv the script gets no packages, so most stop with `ModuleNotFoundError`.
  The rest exit 0 without output.
  `skills/rgpycrumbs-cli/SKILL.md` lists them.
- `eon plt-neb`, and other tools that align periodic structures, install `readcon-ops` from a fixed GitHub commit into `~/.cache/rgpycrumbs/deps` the first time they run.
  It brings `minimage`, a Rust extension built from git, so that run needs the network and `cargo`.
  Each alignment thread calls the installer, so a later run can install it again.
  When that install fails, a `landscape` plot loses the step with only a `Failed to process step` warning and still exits 0.
- `rgpycrumbs eon plt-neb --plot-type landscape` caches its data in `.neb_landscape.parquet` in the current directory and reuses the file for any later run there, even of another band.
  Pass a `--cache-file` per run, or `--force-recompute`.
- Tool scripts declare `requires-python = ">=3.11"` while the package supports 3.10, so `--dev` on Python 3.10 can fail where uv would pick a newer interpreter.
- `uv.lock` is in `.gitignore`, and `pixi.lock` is the committed lock.
- `llms-full.txt` is generated.
  Do not edit it by hand.
  CI fails while it is stale, or while `docs/source/llms.txt` links a page that does not exist.
  Pages missing from that map are printed as notes.
