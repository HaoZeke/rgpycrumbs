#!/usr/bin/env python3
"""Run coverage across pixi environments and combine."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ENVS = {
    "test": ["-m", "pure or eon"],
    "surfaces": ["-m", "surfaces"],
    "fragments": ["-m", "fragments or ira or align"],
    "eonmlflow": ["-m", "eon"],
    "ptm": ["-m", "ptm"],
}


def run_env(env_name, extra_args):
    cov_file = ROOT / f".coverage.{env_name}"
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(cov_file)
    cmd = [
        "pixi", "run", "-e", env_name, "--",
        "python", "-m", "coverage", "run",
        f"--data-file={cov_file}",
        "--source=rgpycrumbs",
        "-m", "pytest", "tests/", "-q",
    ] + extra_args
    print(f"=== {env_name} ===")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    if cov_file.exists():
        print(f"  -> {cov_file} ({cov_file.stat().st_size} bytes)")
    else:
        print(f"  -> NO coverage file produced!")
    return cov_file.exists()


def main():
    # Clean
    for f in ROOT.glob(".coverage*"):
        f.unlink()

    # Run each env
    produced = []
    for env_name, args in ENVS.items():
        if run_env(env_name, args):
            produced.append(str(ROOT / f".coverage.{env_name}"))

    if not produced:
        print("No coverage files produced!")
        sys.exit(1)

    # Combine
    print(f"\n=== Combining {len(produced)} files ===")
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(ROOT / ".coverage")
    subprocess.run(
        ["pixi", "run", "-e", "test", "--",
         "python", "-m", "coverage", "combine"] + produced,
        cwd=ROOT, env=env,
    )
    subprocess.run(
        ["pixi", "run", "-e", "test", "--",
         "python", "-m", "coverage", "report"],
        cwd=ROOT, env=env,
    )


if __name__ == "__main__":
    main()
