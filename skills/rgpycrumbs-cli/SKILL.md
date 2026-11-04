---
name: rgpycrumbs-cli
description: Find, inspect and run rgpycrumbs command-line tools. Covers the script behind each `rgpycrumbs GROUP TOOL` and the Python Enhancement Proposal (PEP) 723 header that decides what uv installs before the tool starts. It also covers the `--dev` and `--verbose` flags, when the dispatcher uses the active environment instead of uv, pinning versions with rgpkgs.toml or a lock file, and that a module without a script metadata block is not a command. Use before running any rgpycrumbs tool, when a tool fails while installing its dependencies, or to find out what a command will download.
license: See LICENSE in the rgpycrumbs repository
compatibility: rgpycrumbs on Python 3.10 or newer (`pip install rgpycrumbs`, or `uv sync` in a checkout). Dispatched tools need Python 3.11 or newer, and uv on PATH unless `--dev` is given. The check scripts need Python 3.11 or newer and the core install only, and run offline.
---

# Running rgpycrumbs tools

## From a command to a script

`rgpycrumbs GROUP TOOL ARGS...` runs the file `rgpycrumbs/GROUP/TOOL.py` of the installed package, with `_` for each `-` in `TOOL`.
`python -m rgpycrumbs.cli` is the same entry point.

| Command | Script |
|---|---|
| `rgpycrumbs eon plt-neb` | `rgpycrumbs/eon/plt_neb.py` |
| `rgpycrumbs uma aoti-cache` | `rgpycrumbs/uma/aoti_cache.py` |
| `rgpycrumbs geom detect-fragments` | `rgpycrumbs/geom/detect_fragments.py` |

`rgpycrumbs --help` lists the groups and `rgpycrumbs GROUP --help` the commands in one group.
Neither installs anything.
A `.py` file directly in a subdirectory is a command only when it starts with a PEP 723 script block.
A name that starts with `_` is never a command.
A subdirectory with at least one command is a group.
`func`, `parsers`, `run`, `search`, `surfaces`, `time` and `xts` hold none, so they are not groups.

`python -m rgpycrumbs.uma.aoti_cache --help` runs one module directly with the current Python, skipping the dispatcher.
Write the module name with `_`.
The dispatcher's help text suggests `python -m rgpycrumbs.uma.aoti-cache`, which does not import.

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

- `rgpycrumbs GROUP TOOL --help` hands `--help` to the script, so the whole dependency list is installed first.
  For the eOn plots that includes jax, matplotlib, polars and ase.
  The pages under <https://rgpycrumbs.rgoswami.me/tools/> list the options without installing anything.
- Most eOn tools, and some in `chemgp`, `geom` and `uma`, take `readcon` and `chemparseplot` from their `main` branches on GitHub through `[tool.uv.sources]`, and the `surrogate` tools take chemparseplot from fixed git commits.
  `scripts/list_tools.py` names them all.
  Building readcon from git needs Rust 1.85 or newer (edition 2024).
  With an older Cargo, uv stops with "feature `edition2024` is required".

## Running in the active environment

- `rgpycrumbs --dev GROUP TOOL ...` runs the script with the current Python.
  Its dependencies must already be installed there.
- Without `--dev`, the dispatcher also uses the current Python when `readcon`, `matplotlib`, `polars`, `ase` and `chemparseplot` all import, or when `uv` is not on PATH, and says so with `--> Using active interpreter`.
  `RGPKGS_FORCE_UV=1`, or `force_uv = true` under `[rgpycrumbs.dispatch]` in a config file, keeps uv in the first case.
  `RGPYCRUMBS_DEV=1` is the same as `--dev`.
  `RGPKGS_DEV=1` is not, and the dispatcher still uses uv.
- `-v` (`--verbose`) prints the script path and the full command before running it.
  Options of `rgpycrumbs` itself go before the group: `rgpycrumbs --dev -v uma aoti-cache list`.

## Pinning versions

Pins become uv constraints for every script the dispatcher runs.
They also reach library imports through `RGPKGS_LOCK_PINS`.
Pins come from a lock file or a config file, highest first:

1. `--lock PATH` (or `--sbom PATH`) on the command line.
2. The `RGPKGS_LOCK` environment variable.
3. A config file named by `RGPKGS_CONFIG`.
4. The project file `rgpkgs.toml`, found by walking up from the current directory (`rgpycrumbs.toml` is still read).
5. The user file `$XDG_CONFIG_HOME/rgpkgs/config.toml`, by default `~/.config/rgpkgs/config.toml`.

A lock is a `uv.lock`, a PEP 751 `pylock.toml` or a CycloneDX JSON file.
`[pins.packages]` entries win over versions from the lock.

```toml
[pins]
lock = "pylock.toml"   # relative to this file

[pins.packages]
jax = "0.4.35"
```

`rgpycrumbs config show` prints the merged result (paths shortened).

```text
user_config:     .../xdg/rgpkgs/config.toml
sources:         .../proj/rgpkgs.toml
lock_path:       .../proj/pylock.toml
force_uv:        False
auto_deps_def:   1
package_pins:    {'jax': '0.4.35'}
```

The same readers are public in `rgpycrumbs.api`.

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

The dispatcher registers no command that lacks a PEP 723 script block.
A module without that block is not a command.
`rgpycrumbs` does not list it and does not run it.
Import the function from the module, for example `from rgpycrumbs.func.muller_brown import muller_brown`.
Other modules in that position include `rgpycrumbs.eon.plot_config` and `rgpycrumbs.surfaces.standard`.

## Check

```bash
python scripts/list_tools.py
python scripts/dispatch_demo.py
```

`scripts/list_tools.py` prints every command with its script, Python requirement, dependency count and git sources.
It fails when a command's script is missing, when the registered commands that lack a script block differ from the names under Commands that are not tools, or when the package directories that are not groups differ from those named at the top.
`scripts/dispatch_demo.py` makes a scratch project with its own `rgpkgs.toml` and `pylock.toml`, runs `rgpycrumbs config show` and `rgpycrumbs --dev -v uma aoti-cache list` there (the one tool that needs nothing beyond click), and checks what they print.
