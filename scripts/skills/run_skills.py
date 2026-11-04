#!/usr/bin/env python3
"""Check every skill under skills/ and run its examples.

A skill is a directory holding a SKILL.md in the Agent Skills format
(https://agentskills.io/specification): YAML frontmatter with ``name``
and ``description``, then Markdown instructions, with helper programs
under ``scripts/``.

For each skill this script

- checks the frontmatter against the specification: the name is 1-64
  lowercase letters, digits and single hyphens and matches the
  directory, the description is 1-1024 characters, the compatibility
  note is at most 500, and no unknown field is present;
- checks that every file in ``scripts/`` is named in SKILL.md, so the
  instructions and the scripts cannot drift apart, and that relative
  links in SKILL.md point at files that exist;
- runs every fenced ``python`` block of SKILL.md (unless its info string
  says ``norun``) and every ``.py`` or ``.sh`` file in ``scripts/``,
  from the skill directory, and fails when one exits non-zero.

    python scripts/skills/run_skills.py              # check and run all
    python scripts/skills/run_skills.py --no-run     # frontmatter and links only
    python scripts/skills/run_skills.py --only NAME  # one skill

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Limits from the specification.
NAME_MAX = 64
DESCRIPTION_MAX = 1024
COMPATIBILITY_MAX = 500
BODY_LINES_ADVISED = 500
NAME_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}

FENCE_RE = re.compile(r"^(?P<indent> *)(?P<ticks>`{3,})(?P<info>[^`]*)$")
LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)\s]+)\)")


def frontmatter(text: str) -> tuple[dict[str, str], list[str], str]:
    """Top-level keys of the YAML frontmatter, problems found, and the body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, ["SKILL.md must start with a '---' frontmatter block"], text
    meta: dict[str, str] = {}
    problems: list[str] = []
    key = None
    for k in range(1, len(lines)):
        line = lines[k]
        if line.strip() == "---":
            return meta, problems, "\n".join(lines[k + 1 :])
        m = re.match(r"([A-Za-z][\w-]*):\s*(.*)$", line)
        if m and not line[0].isspace():
            key, value = m.group(1), m.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if value in {">", "|", ">-", "|-"}:
                value = ""
            meta[key] = value
        elif key is not None and line.strip() and line[0].isspace() and key != "metadata":
            meta[key] = (meta[key] + " " + line.strip()).strip()
    return {}, ["frontmatter block is not closed with '---'"], text


def check_meta(skill: Path, meta: dict[str, str]) -> list[str]:
    problems = []
    name = meta.get("name", "")
    if not name:
        problems.append("missing 'name'")
    else:
        if len(name) > NAME_MAX or not NAME_RE.fullmatch(name):
            problems.append(
                f"name {name!r} must be 1-{NAME_MAX} lowercase letters, digits and single hyphens"
            )
        if name != skill.name:
            problems.append(f"name {name!r} does not match the directory {skill.name!r}")
    description = meta.get("description", "")
    if not description:
        problems.append("missing 'description'")
    elif len(description) > DESCRIPTION_MAX:
        problems.append(
            f"description is {len(description)} characters; the limit is {DESCRIPTION_MAX}"
        )
    compatibility = meta.get("compatibility")
    if compatibility is not None and not 1 <= len(compatibility) <= COMPATIBILITY_MAX:
        problems.append(f"compatibility must be 1-{COMPATIBILITY_MAX} characters")
    unknown = sorted(set(meta) - FIELDS)
    if unknown:
        problems.append("unknown frontmatter fields: " + ", ".join(unknown))
    return problems


def python_blocks(body: str) -> list[tuple[int, str]]:
    """(line number in the body, code) for each fenced python block to run."""
    blocks = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        ticks, words = m.group("ticks"), m.group("info").split()
        indent = len(m.group("indent"))
        j = i + 1
        while j < len(lines) and not (
            lines[j].strip().startswith(ticks) and not lines[j].strip().strip("`")
        ):
            j += 1
        if words and words[0] in {"python", "py"} and "norun" not in words[1:]:
            code = "\n".join(line[indent:] for line in lines[i + 1 : j])
            blocks.append((i + 1, code + "\n"))
        i = j + 1
    return blocks


def check_files(skill: Path, body: str) -> list[str]:
    problems = []
    scripts = skill / "scripts"
    if scripts.is_dir():
        for path in sorted(
            p for p in scripts.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        ):
            rel = path.relative_to(skill).as_posix()
            if rel not in body:
                problems.append(f"{rel} is not named in SKILL.md")
    for m in LINK_RE.finditer(body):
        target = m.group("target").split("#", 1)[0]
        if not target or re.match(r"[A-Za-z][\w+.-]*:", target):
            continue
        if not (skill / target).exists():
            problems.append(f"link target {target!r} does not exist")
    return problems


def run(cmd: list[str], cwd: Path, label: str) -> bool:
    print(f"--- {label}", flush=True)
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        print(f"FAILED ({label}): exit status {proc.returncode}", flush=True)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skills-dir", type=Path, default=ROOT / "skills")
    parser.add_argument(
        "--only", action="append", default=[], help="skill name to check (repeatable)"
    )
    parser.add_argument(
        "--no-run", action="store_true", help="check files without running examples"
    )
    args = parser.parse_args(argv)

    skills = sorted(p.parent for p in args.skills_dir.glob("*/SKILL.md"))
    if args.only:
        skills = [s for s in skills if s.name in args.only]
    if not skills:
        print(f"no skills found under {args.skills_dir}", file=sys.stderr)
        return 1
    failures = 0
    for skill in skills:
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        meta, problems, body = frontmatter(text)
        problems += check_meta(skill, meta) if meta else []
        problems += check_files(skill, body)
        lines = len(text.splitlines())
        print(f"=== {skill.name} ({lines} lines)", flush=True)
        if lines >= BODY_LINES_ADVISED:
            print(
                f"note: SKILL.md has {lines} lines; the specification advises fewer than {BODY_LINES_ADVISED}"
            )
        for problem in problems:
            print(f"FAILED ({skill.name}): {problem}")
        failures += len(problems)
        if args.no_run or problems:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            for number, code in python_blocks(body):
                path = Path(tmp) / f"block_{number}.py"
                path.write_text(code, encoding="utf-8")
                failures += not run(
                    [sys.executable, str(path)],
                    skill,
                    f"SKILL.md python block at body line {number}",
                )
        scripts = skill / "scripts"
        for path in sorted(scripts.glob("*")) if scripts.is_dir() else []:
            if path.suffix == ".py":
                failures += not run(
                    [sys.executable, str(path)], skill, path.relative_to(skill).as_posix()
                )
            elif path.suffix == ".sh":
                failures += not run(
                    ["bash", str(path)], skill, path.relative_to(skill).as_posix()
                )
    print(f"=== {len(skills)} skill(s), {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
