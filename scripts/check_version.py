#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail when a version copy disagrees with the git tag.

hatch-vcs derives the package version from the latest ``vX.Y.Z`` tag, so no
tracked file carries its own version field. The towncrier heading is the only
version typed at release time. It must match the latest tag reachable from
HEAD. The one exception is the release commit itself: when the subject of HEAD
is ``release: vX.Y.Z`` for the newest heading, that heading may run ahead of
the tag, until the tag points at HEAD. It must then be newer than the
latest tag, and equal to the next version from ``cog bump --dry-run --auto``.
That check needs cog on PATH and a branch listed in cog.toml; when cog is
missing, fails, or prints no version, the release commit fails too. An
older ``release:`` commit never stands in for a missing tag.

Usage: python scripts/check_version.py [EXPECTED]
"""

import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
HEADING = re.compile(r"^(?:rgpycrumbs )?(\d+\.\d+\.\d+) \(\d{4}-\d{2}-\d{2}\)$", re.M)
VERSION = re.compile(r"v?(\d+\.\d+\.\d+)")
COG_INSTALL = "cargo install --locked cocogitto@7.0.0"


def run(*args):
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git(*args):
    return run("git", *args)


def latest_tag():
    try:
        return git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*")
    except subprocess.CalledProcessError:
        return None


def cog_next_version():
    """Return ``(version, None)`` from cog, or ``(None, error)``."""
    try:
        out = run("cog", "bump", "--dry-run", "--auto")
    except FileNotFoundError:
        return None, f"cog is not on PATH; install it with {COG_INSTALL}"
    except subprocess.CalledProcessError as err:
        reason = " ".join((err.stderr or err.stdout or "").split())
        reason = reason or f"exit {err.returncode}"
        return None, f"cog bump --dry-run --auto failed: {reason}"
    match = VERSION.fullmatch(out)
    if not match:
        return None, f"cog bump --dry-run --auto printed {out!r}, not a version"
    return match.group(1), None


def version_key(version):
    return tuple(int(part) for part in version.split("."))


def main(argv):
    errors = []
    with (ROOT / "pixi.toml").open("rb") as f:
        workspace = tomllib.load(f).get("workspace", {})
    if "version" in workspace:
        errors.append(f"pixi.toml sets workspace.version = {workspace['version']!r}")

    match = HEADING.search((ROOT / "CHANGELOG.rst").read_text())
    heading = match.group(1) if match else None
    tag = latest_tag()
    tagged = tag.removeprefix("v") if tag else None
    if argv:
        expected = argv[0].removeprefix("v")
    elif (
        heading
        and git("log", "-1", "--format=%s") == f"release: v{heading}"
        and not git("tag", "--points-at", "HEAD", "--list", "v[0-9]*")
    ):
        if tagged and version_key(heading) <= version_key(tagged):
            errors.append(f"release heading {heading} is not newer than {tag}")
        expected, cog_error = cog_next_version()
        if cog_error:
            errors.append(cog_error)
    else:
        expected = tagged
        if expected is None:
            errors.append("no v* tag is reachable from HEAD")
    if expected is not None and heading != expected:
        errors.append(f"CHANGELOG.rst heading is {heading}, expected {expected}")

    for error in errors:
        print(f"version drift: {error}", file=sys.stderr)
    if not errors:
        print(f"version {expected}: all copies agree")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
