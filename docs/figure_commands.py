"""Render tutorial figures by running the commands printed on the pages.

Each command is written exactly as it appears in its org page, with ``-o``
naming the bare image file, and ``-o`` is redirected into the matching
``docs/source/_static`` folder. Every command runs in a fresh empty folder whose
``data`` entry links to ``docs/orgmode/tutorials/data``. The relative data paths
on the pages resolve there, and no plot cache from an earlier command or run is
read back.
"""

import importlib
import os
import re
import shlex
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
from click.testing import CliRunner

DOCS = Path(__file__).resolve().parent
TUTORIALS = DOCS / "orgmode" / "tutorials"
MODULES = {
    "plt-neb": "rgpycrumbs.eon.plt_neb",
    "plt-saddle": "rgpycrumbs.eon.plt_saddle",
    "plt-min": "rgpycrumbs.eon.plt_min",
}
NEB_INPUTS = (
    '--input-dat-pattern "data/diels_alder/neb_*.dat" '
    '--input-path-pattern "data/diels_alder/neb_path_*.con" '
    "--con-file data/diels_alder/neb.con --sp-file data/diels_alder/sp.con"
)


def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\\\n", " "))


def render(pages: dict[str, list[str]], out_dir: Path) -> None:
    """Run every ``rgpycrumbs eon <tool>`` command and save its figure.

    ``pages`` maps each org page to the commands it prints. The old image is
    deleted first, so a failed command cannot leave a stale file behind.
    After the last command, exits non-zero if any command failed or does not
    appear verbatim on its page.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RGPYCRUMBS_SUPPRESS_SCRIPT_IMPORT_WARNING", "1")
    runner = CliRunner()
    failed = []
    for page, commands in pages.items():
        text = _flatten((TUTORIALS / page).read_text())
        for command in commands:
            program, group, tool, *args = shlex.split(command)
            if (program, group) != ("rgpycrumbs", "eon") or tool not in MODULES:
                raise SystemExit(f"not an rgpycrumbs eon plot command: {command}")
            slot = args.index("-o") + 1
            target = out_dir / args[slot]
            target.unlink(missing_ok=True)
            args[slot] = str(target)
            cli = importlib.import_module(MODULES[tool]).main
            with runner.isolated_filesystem():
                Path("data").symlink_to(TUTORIALS / "data")
                result = runner.invoke(cli, args)
            plt.close("all")
            on_page = _flatten(command) in text
            ok = result.exit_code == 0 and target.is_file() and on_page
            if ok:
                print(f"OK: {command}")
            else:
                print(f"FAIL (exit {result.exit_code}, on {page}: {on_page}): {command}")
                failed.append(target.name)
            if result.exception is not None and result.exit_code != 0:
                traceback.print_exception(result.exception)
    if failed:
        raise SystemExit(f"{len(failed)} figure(s) failed: {', '.join(failed)}")
