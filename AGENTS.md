# AGENTS.md

Working notes for changing or scripting against rgpycrumbs. The manual
is Org under `docs/orgmode/`, published at
<https://rgpycrumbs.rgoswami.me/>. `docs/source/llms.txt` is its
hand-written map, served as <https://rgpycrumbs.rgoswami.me/llms.txt>,
and `llms-full.txt` at the repository root holds the whole manual in one
file. Task guides with scripts that CI runs are under `skills/`.

## What rgpycrumbs is

- A command dispatcher. `rgpycrumbs GROUP TOOL ARGS` runs the script
  `rgpycrumbs/GROUP/TOOL.py` (dashes in `TOOL` read as underscores)
  under `uv run --script`, so each tool installs what its PEP 723 header
  names before it starts. The groups are `chemgp`, `eon`, `func`,
  `geom`, `orca`, `parsers`, `plumed`, `prefix`, `run`, `surfaces`,
  `surrogate` and `uma`; `rgpycrumbs config show` prints the merged
  configuration.
- A library. `docs/orgmode/explanation/public_api.org` lists what is
  public, mostly `rgpycrumbs.api`. `import rgpycrumbs` needs only click,
  numpy and rich. `rgpycrumbs.surfaces`, `.interpolation` and `.geom`
  load on first use and raise `ImportError` when their packages (jax,
  scipy, ase) are missing.

## Layout

| Path | Contents |
|---|---|
| `rgpycrumbs/cli.py` | The dispatcher: command discovery, pins, `uv run` |
| `rgpycrumbs/config.py`, `locks.py` | `rgpkgs.toml` layering, lock and SBOM readers |
| `rgpycrumbs/_aux.py` | `ensure_import`, runtime installs into `~/.cache/rgpycrumbs/deps` |
| `rgpycrumbs/GROUP/` | One directory per command group; tools and library modules |
| `tests/` | pytest suite, split by markers (see Test) |
| `docs/orgmode/` | Manual sources (Org); `docs/source/` is the Sphinx side |
| `docs/orgmode/tutorials/data/` | Sample eOn runs the tutorials and skills read |
| `docs/newsfragments/` | towncrier fragments for the next release (created on demand) |
| `skills/` | Task guides, each a `SKILL.md` plus `scripts/` |
| `scripts/` | Org doctest runner, README export, `llms/`, `skills/` |
| `readme_src.org` | Source of `README.md` |

## Install

```bash
uv sync --extra test          # Python 3.10 or newer; CI uses --all-extras
uv run rgpycrumbs --help
```

Dispatched tools need Python 3.11 or newer and `uv` on `PATH`. For the
suites that need conda packages (ira, tblite, ovito, eOn) use pixi; its
environments are `test`, `surfaces`, `fragments`, `eonmlflow`, `ptm`,
`docs` and `bench`, for `linux-64` only.

## Test

These are the CI jobs in `.github/workflows/ci_test.yml`,
`ci_doctest.yml` and `ci_tutorials.yml`:

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

The markers are declared in `pyproject.toml`: `pure`, `interpolation`,
`align`, `surfaces`, `fragments`, `eon`, `ptm`, `ira` and `slow`. A
module whose packages are missing skips itself (`skip_if_not_env` in
`tests/conftest.py`), so add `-rs` to see what did not run. The PEP 723
check fails when a tool imports a package its header does not list.

and in `.github/workflows/ci_skills.yml`:

```bash
python scripts/llms/build_llms.py --check   # llms-full.txt is current
python -m unittest discover -s scripts/llms -p "test_*.py"   # the generator
python scripts/skills/run_skills.py --no-run   # every SKILL.md is well formed
uv run python scripts/skills/run_skills.py --only rgpycrumbs-cli --only rgpycrumbs-add-tool
```

The first three need Python 3.11 or newer and nothing else. The NEB
skill needs the plot stack in the running environment; the workflow
shows the `uv run --with ...` line.

## Lint

```bash
uvx prek run -a
```

runs ruff (line length 90, Python 3.10 target), ruff-format, taplo,
codespell and the YAML and TOML checks from `prek.toml`. When
`readme_src.org` changes and Emacs is installed, it also regenerates
`README.md`.

## Docs

```bash
pixi r docbld
```

runs `genfigs` in the `test` environment, exports the Org files to
`docs/source/*.rst` with `emacs --batch --load export.el` (from `docs/`;
it fetches `ox-rst` from MELPA on first use), then
`sphinx-build docs/source docs/build`. The `.rst` files are generated
and ignored by git; edit the Org files.

## Common tasks

- Run a tool, find out what it installs, or pin versions:
  `skills/rgpycrumbs-cli/`.
- Barriers and figures from an eOn NEB run: `skills/rgpycrumbs-neb-plots/`.
- Add a tool: `skills/rgpycrumbs-add-tool/`.
- Edit the manual: change the `.org` file; a new page also goes in a
  toctree (inside a `#+BEGIN_EXPORT rst` block) and in
  `docs/source/llms.txt`. Then run `python scripts/llms/build_llms.py`
  (settings in `scripts/llms/llms.toml`) to refresh `llms-full.txt`.
- Edit the README: change `readme_src.org`, then
  `bash scripts/org_to_md.sh readme_src.org README.md` (Emacs).
- Record a change: a fragment `docs/newsfragments/NAME.TYPE.rst`, one
  paragraph of reStructuredText, where `TYPE` is `security`, `removed`,
  `deprecated`, `added`, `dev`, `changed`, `fixed` or `misc`; prefix
  `NAME` with `+` when there is no issue number. Add it with
  `git add -f`, since `.gitignore` has `*.rst`. `CHANGELOG.rst` is
  written by `towncrier build` at release time; do not edit it.
- Commit subjects are conventional (`type(scope): subject`) with
  cocogitto's default types; `cog.toml` adds none, so write `docs`, not
  `doc`. `docs/orgmode/contributing/release.org` covers releases.

## Pitfalls

- The version comes from the git tag through hatch-vcs. There is no
  version to edit; the `version = "1.8.0"` in `pixi.toml` is unused and
  stale. In a checkout of v1.13.0, `rgpycrumbs.__version__` is
  `unknown`, because hatch-vcs writes `_version.py` at the repository
  root; `rgpycrumbs --version` and
  `importlib.metadata.version("rgpycrumbs")` give the real one.
- `rgpycrumbs GROUP TOOL --help` passes `--help` to the script, so uv
  installs the tool's whole dependency list first. The pages under
  <https://rgpycrumbs.rgoswami.me/tools/> list the options for free.
- Many tool headers take readcon and chemparseplot from git through
  `[tool.uv.sources]`. Building readcon from git needs Rust 1.85 or
  newer; with an older Cargo, uv stops with "feature `edition2024` is
  required". Installing the PyPI releases into the running environment
  avoids the build (`skills/rgpycrumbs-neb-plots/`).
- The dispatcher runs a tool with the current Python, not uv, when
  readcon, matplotlib, polars, ase and chemparseplot all import, or when
  uv is missing. The tool then uses the versions installed there, not
  those its header asks for; `RGPKGS_FORCE_UV=1` forces uv.
  `RGPYCRUMBS_DEV=1` acts as `--dev`, but `RGPKGS_DEV=1` does not.
- Sixteen listed commands, such as `geom analysis` and `func
  muller-brown`, are library modules and do no work when run. Under uv
  the script gets no packages, so most stop with `ModuleNotFoundError`;
  the rest exit 0 without output. `skills/rgpycrumbs-cli/SKILL.md`
  lists them.
- `eon plt-neb`, and other tools that align periodic structures,
  install `readcon-ops` from a fixed GitHub commit into
  `~/.cache/rgpycrumbs/deps` the first time they run. It brings
  `minimage`, a Rust extension built from git, so that run needs the
  network and `cargo`. The alignment runs on threads that each call the
  installer, so a later run can install it again; when that fails, a
  landscape loses the step with only a `Failed to process step` warning
  and still exits 0.
- `rgpycrumbs eon plt-neb --plot-type landscape` caches its data in
  `.neb_landscape.parquet` in the current directory and reuses the file
  for any later run there, even of another band. Pass a `--cache-file`
  per run, or `--force-recompute`.
- Tool scripts declare `requires-python = ">=3.11"` while the package
  supports 3.10, so `--dev` on Python 3.10 can fail where uv would pick
  a newer interpreter.
- `uv.lock` is in `.gitignore`; `pixi.lock` is the committed lock.
- `llms-full.txt` is generated. Do not edit it by hand. CI fails while
  it is stale, or while `docs/source/llms.txt` links a page that does
  not exist; pages missing from that map are printed as notes.
