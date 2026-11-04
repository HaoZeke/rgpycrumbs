"""List every rgpycrumbs command, the script it runs and what it installs.

Fails when a command's script is missing, or when the commands without
a PEP 723 block differ from the list in SKILL.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

from rgpycrumbs.cli import PACKAGE_ROOT, main

# The reference regular expression from PEP 723.
BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)
SKILL = Path(__file__).resolve().parents[1] / "SKILL.md"


def script_header(path: Path) -> dict | None:
    for m in BLOCK.finditer(path.read_text(encoding="utf-8")):
        if m.group("type") == "script":
            lines = m.group("content").splitlines(keepends=True)
            body = "".join(ln[2:] if ln.startswith("# ") else ln[1:] for ln in lines)
            return tomllib.loads(body)
    return None


def git_requirements(header: dict) -> list[str]:
    """Packages taken from git, as ``name@branch`` or ``name@commit``."""
    found = []
    sources = header.get("tool", {}).get("uv", {}).get("sources", {})
    for name, source in sorted(sources.items()):
        if "git" in source:
            ref = source.get("branch") or source.get("tag") or source.get("rev", "")
            found.append(f"{name}@{ref[:12]}")
    for dep in header.get("dependencies", []):
        m = re.match(r"([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?\s*@\s*git\+[^@]+@(\S+)", dep)
        if m:
            found.append(f"{m.group(1)}@{m.group(2)[:12]}")
    return found


def documented_non_tools() -> set[str]:
    text = SKILL.read_text(encoding="utf-8")
    section = text.split("## Commands that are not tools", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"`([a-z]+ [a-z0-9-]+)`", section))


def documented_idle_dirs() -> set[str]:
    """Package directories that SKILL.md says hold no command."""
    text = " ".join(SKILL.read_text(encoding="utf-8").split())
    m = re.search(r"((?:`\w+`(?:, | and ))+`\w+`) hold none", text)
    return set(re.findall(r"`(\w+)`", m.group(1))) if m else set()


def run() -> int:
    failures = 0
    dirs = {
        d.name
        for d in PACKAGE_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    }
    groups = {name for name, cmd in main.commands.items() if hasattr(cmd, "commands")}
    idle = dirs - groups
    print(f"directories that are not groups: {', '.join(sorted(idle))}")
    if idle != documented_idle_dirs():
        failures += 1
        print("FAILED: SKILL.md names the wrong directories as holding no command")
    non_tools = set()
    for group_name, group in sorted(main.commands.items()):
        if not hasattr(group, "commands"):
            continue
        for name in sorted(group.commands):
            path = PACKAGE_ROOT / group_name / f"{name.replace('-', '_')}.py"
            command = f"{group_name} {name}"
            if not path.is_file():
                print(f"FAILED: {command} has no script at {path}")
                failures += 1
                continue
            header = script_header(path)
            rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
            if header is None:
                non_tools.add(command)
                print(f"{command:36} {rel}  (no PEP 723 block: not a tool)")
                continue
            git = git_requirements(header)
            deps = header.get("dependencies", [])
            python = header.get("requires-python", "any")
            line = f"{command:36} {rel}  python{python}  {len(deps)} deps"
            print(line + (f"  git: {', '.join(git)}" if git else ""))
    documented = documented_non_tools()
    if non_tools != documented:
        failures += 1
        print("FAILED: SKILL.md lists the wrong commands under 'not tools'")
        for command in sorted(non_tools - documented):
            print(f"  missing from SKILL.md: {command}")
        for command in sorted(documented - non_tools):
            print(f"  listed in SKILL.md but a tool or gone: {command}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
