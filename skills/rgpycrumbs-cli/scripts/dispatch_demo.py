"""Run `rgpycrumbs config show` and a --dev dispatch in a scratch project.

The project has its own rgpkgs.toml and pylock.toml, and the user
config and caches point into the same temporary directory, so nothing
outside it is read or written.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PYLOCK = """\
lock-version = "1.0"
created-by = "example"

[[packages]]
name = "numpy"
version = "2.0.2"

[[packages]]
name = "jax"
version = "0.4.31"
"""

CONFIG = """\
[pins]
lock = "pylock.toml"

[pins.packages]
jax = "0.4.35"
"""


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
    return proc.stdout + proc.stderr


def run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        proj = root / "proj"
        proj.mkdir()
        (proj / "pylock.toml").write_text(PYLOCK, encoding="utf-8")
        (proj / "rgpkgs.toml").write_text(CONFIG, encoding="utf-8")
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("RGPKGS_", "RGPYCRUMBS_"))
        }
        env["XDG_CONFIG_HOME"] = str(root / "xdg")
        env["XDG_CACHE_HOME"] = str(root / "cache")

        shown = rgpycrumbs("config", "show", cwd=proj, env=env)
        assert f"sources:         {proj / 'rgpkgs.toml'}" in shown, shown
        assert f"lock_path:       {proj / 'pylock.toml'}" in shown, shown
        assert "package_pins:    {'jax': '0.4.35'}" in shown, shown

        uma = root / "uma"
        out = rgpycrumbs(
            "--dev", "-v", "uma", "aoti-cache", "list", "--cache", str(uma),
            cwd=proj,
            env=env,
        )  # fmt: skip
        assert "2 PyPI pin(s) for constraints" in out, out
        assert "uma/aoti_cache.py list --cache" in out, out
        assert f"cache {uma}: 0 package(s)" in out, out
    print("ok")


if __name__ == "__main__":
    run()
