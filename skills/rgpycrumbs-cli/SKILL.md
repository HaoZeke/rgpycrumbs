---
name: rgpycrumbs-cli
description: Find, inspect and run rgpycrumbs command-line tools. Covers how `rgpycrumbs GROUP TOOL` maps to a script, how that script's PEP 723 header decides what uv installs before the tool starts, the --dev and --verbose flags, when the active environment is used instead of uv, pinning versions with rgpkgs.toml or a lock file, and the listed commands that are library modules rather than tools. Use before running any rgpycrumbs tool, when a tool fails while installing its dependencies, or to find out what a command will download.
license: MIT
compatibility: rgpycrumbs on Python 3.10 or newer (pip install rgpycrumbs, or uv sync in a checkout). Dispatched tools need Python 3.11 or newer, and uv on PATH unless --dev is given. The check scripts need Python 3.11 or newer and the core install only, and run offline.
---

# Running rgpycrumbs tools

## From a command to a script

`rgpycrumbs GROUP TOOL ARGS...` runs the file `rgpycrumbs/GROUP/TOOL.py` of the installed package, reading dashes in `TOOL` as underscores.
`python -m rgpycrumbs.cli` is the same entry point.

| Command | Script |
|---|---|
| `rgpycrumbs eon plt-neb` | `rgpycrumbs/eon/plt_neb.py` |
| `rgpycrumbs uma aoti-cache` | `rgpycrumbs/uma/aoti_cache.py` |
| `rgpycrumbs geom detect-fragments` | `rgpycrumbs/geom/detect_fragments.py` |

`rgpycrumbs --help` lists the groups and `rgpycrumbs GROUP --help` the commands in one group; neither installs anything.
Every `.py` file directly in a subdirectory of the package is a command, unless its name starts with `_` or is `hdf5_io`, `plotting`, `utils`, `helpers` or `seed_dimers`, and a subdirectory with at least one command is a group.
`search`, `time` and `xts` hold none, so they are not groups.

`python -m rgpycrumbs.uma.aoti_cache --help` runs one module directly with the current Python, skipping the dispatcher.
Spell the module with underscores: the help text of the dispatcher's commands suggests `python -m rgpycrumbs.uma.aoti-cache`, which does not import.

## What a tool installs

A tool script starts with a PEP 723 block, for example:

```text
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "chemparseplot[neb,plot]>=1.9.17,<2",
#   "readcon>=0.17.1",
# ]
#
# [tool.uv.sources]
# readcon = { git = "https://github.com/lode-org/readcon-core", branch = "main" }
# ///
```

The dispatcher copies the script to a temporary directory and runs `uv run --no-project --script COPY ARGS`.
uv builds an environment holding exactly those packages, caches it, and only then starts the tool.
Two things follow:

- `rgpycrumbs GROUP TOOL --help` hands `--help` to the script, so the whole dependency list is installed first; for the eOn plots that includes jax, matplotlib, polars and ase.
  The pages under <https://rgpycrumbs.rgoswami.me/tools/> list the options without installing anything.
- Most eOn tools, and some in `chemgp`, `geom` and `uma`, take `readcon` and `chemparseplot` from their `main` branches on GitHub through `[tool.uv.sources]`, and the `surrogate` tools take chemparseplot from fixed git commits; `scripts/list_tools.py` names them all.
  Building readcon from git needs Rust 1.85 or newer (edition 2024).
  With an older Cargo, uv stops with "feature `edition2024` is required".

## Running in the active environment

- `rgpycrumbs --dev GROUP TOOL ...` runs the script with the current Python, so its dependencies must already be installed there.
- Without `--dev`, the dispatcher also uses the current Python when `readcon`, `matplotlib`, `polars`, `ase` and `chemparseplot` all import, or when `uv` is not on PATH, and says so with `--> Using active interpreter`.
  `RGPKGS_FORCE_UV=1`, or `force_uv = true` under `[rgpycrumbs.dispatch]` in a config file, keeps uv in the first case.
  `RGPYCRUMBS_DEV=1` is the same as `--dev`; `RGPKGS_DEV=1` is not, and the dispatcher still uses uv.
- `-v` (`--verbose`) prints the script path and the full command before running it.
  Options of `rgpycrumbs` itself go before the group: `rgpycrumbs --dev -v uma aoti-cache list`.

## Pinning versions

Pins become uv constraints for every script the dispatcher runs, and are passed to library imports through `RGPKGS_LOCK_PINS`.
They come from, highest first:

1. `--lock PATH` (or `--sbom PATH`) on the command line;
2. the `RGPKGS_LOCK` environment variable;
3. a config file named by `RGPKGS_CONFIG`;
4. the project file `rgpkgs.toml`, found by walking up from the current directory (`rgpycrumbs.toml` is still read);
5. the user file `$XDG_CONFIG_HOME/rgpkgs/config.toml`, by default `~/.config/rgpkgs/config.toml`.

A lock is a `uv.lock`, a PEP 751 `pylock.toml` or a CycloneDX JSON file.
`[pins.packages]` entries win over versions from the lock:

```toml
[pins]
lock = "pylock.toml"   # relative to this file

[pins.packages]
jax = "0.4.35"
```

`rgpycrumbs config show` prints the merged result (paths shortened):

```text
user_config:     .../xdg/rgpkgs/config.toml
sources:         .../proj/rgpkgs.toml
lock_path:       .../proj/pylock.toml
force_uv:        False
auto_deps_def:   1
package_pins:    {'jax': '0.4.35'}
```

The same readers are public in `rgpycrumbs.api`:

```python
import tempfile
from pathlib import Path

from rgpycrumbs.api import apply_pin_to_spec, load_pypi_pins, pins_to_constraint_lines

LOCK = """\
lock-version = "1.0"
created-by = "example"

[[packages]]
name = "numpy"
version = "2.0.2"

[[packages]]
name = "jax"
version = "0.4.31"
"""

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "pylock.toml"
    path.write_text(LOCK, encoding="utf-8")
    pins = load_pypi_pins(path)

assert pins == {"numpy": "2.0.2", "jax": "0.4.31"}
assert pins_to_constraint_lines(pins) == ["jax==0.4.31", "numpy==2.0.2"]
assert apply_pin_to_spec("jax>=0.4", pins) == "jax==0.4.31"
assert apply_pin_to_spec("ase>=3.22", pins) == "ase>=3.22"
print(pins_to_constraint_lines(pins))
```

## Commands that are not tools

These commands are library modules, with no PEP 723 block and no `__main__` section, so running one does no work.
Under uv the script gets no packages, and those that import numpy, jax or any other package stop with `ModuleNotFoundError`; the others, and any of them run with `--dev` where its imports are installed, exit 0 without output.
Import them from Python instead (`from rgpycrumbs.func.muller_brown import muller_brown`).

- `eon plot-config`, `eon plot-dispatch`
- `func muller-brown`
- `geom analysis`, `geom fragment-visualization`, `geom fragments`, `geom ira`
- `parsers bless`, `parsers common`
- `run jupyter`
- `surfaces gradient-imq`, `surfaces gradient-matern`, `surfaces gradient-nystrom`, `surfaces gradient-rq`, `surfaces gradient-se`, `surfaces standard`

## Check

```bash
python scripts/list_tools.py
python scripts/dispatch_demo.py
```

`scripts/list_tools.py` prints every command with its script, Python requirement, dependency count and git sources.
It fails when a command's script is missing, when the commands without a PEP 723 block differ from the list above, or when the package directories that are not groups differ from those named at the top.
`scripts/dispatch_demo.py` makes a scratch project with its own `rgpkgs.toml` and `pylock.toml`, runs `rgpycrumbs config show` and `rgpycrumbs --dev -v uma aoti-cache list` there (the one tool that needs nothing beyond click), and checks what they print.
