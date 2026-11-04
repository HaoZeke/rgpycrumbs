"""Add a tool to a scratch copy of rgpycrumbs and check how it is dispatched.

The installed package is copied into a temporary directory, which goes
first on PYTHONPATH, so the real checkout is never touched.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

from rgpycrumbs.cli import PACKAGE_ROOT

TEMPLATE = Path(__file__).resolve().parents[1] / "assets" / "count_atoms.py"
BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)
WATER = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n"


def rgpycrumbs(*args: str, cwd: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "rgpycrumbs.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"$ rgpycrumbs {' '.join(args)}")
    print(proc.stdout + proc.stderr, end="")
    if proc.returncode != 0:
        sys.exit(f"exit status {proc.returncode}")
    return proc.stdout


def run() -> None:
    m = next(b for b in BLOCK.finditer(TEMPLATE.read_text(encoding="utf-8")))
    lines = m.group("content").splitlines(keepends=True)
    header = tomllib.loads(
        "".join(ln[2:] if ln.startswith("# ") else ln[1:] for ln in lines)
    )
    assert header["dependencies"] == ["click>=8.1"], header
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        package = root / "pkg" / "rgpycrumbs"
        shutil.copytree(
            PACKAGE_ROOT, package, ignore=shutil.ignore_patterns("__pycache__")
        )
        group = package / "demo"
        group.mkdir()
        (group / "__init__.py").write_text("", encoding="utf-8")
        shutil.copy2(TEMPLATE, group / "count_atoms.py")
        (group / "_private.py").write_text("VALUE = 1\n", encoding="utf-8")
        (group / "plotting.py").write_text("VALUE = 2\n", encoding="utf-8")
        (group / "analysis.py").write_text("VALUE = 3\n", encoding="utf-8")
        (root / "water.xyz").write_text(WATER, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root / "pkg"), *filter(None, [env.get("PYTHONPATH")])]
        )

        listing = rgpycrumbs("demo", "--help", cwd=root, env=env)
        commands = listing.split("Commands:", 1)[1].split()
        assert "count-atoms" in commands, listing
        assert "analysis" not in commands, listing
        assert "private" not in listing, listing
        assert "plotting" not in commands, listing

        counted = rgpycrumbs(
            "--dev", "demo", "count-atoms", "water.xyz", cwd=root, env=env
        )
        assert counted.splitlines()[-2:] == ["H 2", "O 1"], counted
    print("ok")


if __name__ == "__main__":
    run()
