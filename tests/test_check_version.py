# SPDX-License-Identifier: MIT
"""scripts/check_version.py against throwaway git histories."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_version.py"
RELEASE_DATE = "2026-01-01"
REAL_COG = shutil.which("cog")
GIT_ENV = {
    "GIT_AUTHOR_NAME": "check-version",
    "GIT_AUTHOR_EMAIL": "",
    "GIT_COMMITTER_NAME": "check-version",
    "GIT_COMMITTER_EMAIL": "",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
# Stands in for cog: prints $NEXT_VERSION, or fails like cog off a bump branch.
COG_STUB = """#!/bin/sh
if [ -z "$NEXT_VERSION" ]; then
  echo "Error: bump is not allowed" >&2
  exit 1
fi
echo "$NEXT_VERSION"
"""
NO_COG = "cog is not on PATH; install it with cargo install --locked cocogitto@7.0.0"


def git(repo, *args):
    env = {**os.environ, **GIT_ENV}
    subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )


def release(repo, version):
    """Prepend a towncrier heading for *version* and commit it as a release."""
    title = f"rgpycrumbs {version} ({RELEASE_DATE})"
    changelog = repo / "CHANGELOG.rst"
    changelog.write_text(f"{title}\n{'=' * len(title)}\n\n{changelog.read_text()}")
    git(repo, "commit", "-q", "-am", f"release: v{version}")


def check(repo, *args, next_version="", cog="stub"):
    """Run the script with only git on PATH, plus the stub or the real cog."""
    path = [str(repo.parent / "git-only")]
    if cog == "stub":
        path.insert(0, str(repo.parent / "bin"))
    elif cog == "real":
        path.insert(0, str(Path(REAL_COG).parent))
    env = {
        **os.environ,
        **GIT_ENV,
        "NEXT_VERSION": next_version,
        "PATH": os.pathsep.join(path),
    }
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_version.py"), *args],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path):
    stub = tmp_path / "bin" / "cog"
    stub.parent.mkdir()
    stub.write_text(COG_STUB)
    stub.chmod(0o755)
    (tmp_path / "git-only").mkdir()
    (tmp_path / "git-only" / "git").symlink_to(shutil.which("git"))
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / "check_version.py")
    (repo / "pixi.toml").write_text('[workspace]\nname = "pycrumbs"\n')
    (repo / "CHANGELOG.rst").write_text("")
    shutil.copy(ROOT / "cog.toml", repo / "cog.toml")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", ".")
    release(repo, "1.0.0")
    git(repo, "tag", "v1.0.0")
    return repo


@pytest.fixture
def repo_1_13(repo):
    """The v1.13.0 history from the review: cog tags 1.13.1 next."""
    release(repo, "1.13.0")
    git(repo, "tag", "v1.13.0")
    git(repo, "commit", "-q", "--allow-empty", "-m", "fix: patch after v1.13.0")
    return repo


# Release headings on top of v1.13.0, and the errors when cog prints v1.13.1.
REVIEW_HEADINGS = {
    "1.7.1": [
        "release heading 1.7.1 is not newer than v1.13.0",
        "CHANGELOG.rst heading is 1.7.1, expected 1.13.1",
    ],
    "1.13.0": [
        "release heading 1.13.0 is not newer than v1.13.0",
        "CHANGELOG.rst heading is 1.13.0, expected 1.13.1",
    ],
    "1.14.0": ["CHANGELOG.rst heading is 1.14.0, expected 1.13.1"],
    "1.13.1": [],
}


def drift(errors):
    return [f"version drift: {error}" for error in errors]


def test_tagged_release_passes(repo):
    result = check(repo)
    assert result.returncode == 0, result.stderr
    assert "version 1.0.0" in result.stdout


@pytest.mark.parametrize(("heading", "errors"), REVIEW_HEADINGS.items())
def test_review_headings_with_cog(repo_1_13, heading, errors):
    release(repo_1_13, heading)
    result = check(repo_1_13, next_version="v1.13.1")
    assert result.stderr.splitlines() == drift(errors)
    assert result.returncode == (1 if errors else 0)


@pytest.mark.parametrize(("heading", "errors"), REVIEW_HEADINGS.items())
def test_review_headings_without_cog_fail(repo_1_13, heading, errors):
    release(repo_1_13, heading)
    result = check(repo_1_13, cog="absent")
    newer = [error for error in errors if "not newer" in error]
    assert result.stderr.splitlines() == drift([*newer, NO_COG])
    assert result.returncode == 1
    assert result.stdout == ""


@pytest.mark.skipif(REAL_COG is None, reason="cog is not on PATH")
@pytest.mark.parametrize(("heading", "errors"), REVIEW_HEADINGS.items())
def test_review_headings_with_real_cog(repo_1_13, heading, errors):
    release(repo_1_13, heading)
    result = check(repo_1_13, cog="real")
    assert result.stderr.splitlines() == drift(errors)
    assert result.returncode == (1 if errors else 0)


@pytest.mark.skipif(REAL_COG is None, reason="cog is not on PATH")
def test_real_cog_off_a_bump_branch_fails(repo_1_13):
    git(repo_1_13, "checkout", "-q", "-b", "topic")
    release(repo_1_13, "1.13.1")
    result = check(repo_1_13, cog="real")
    assert result.returncode == 1
    assert "cog bump --dry-run --auto failed: " in result.stderr
    assert "topic" in result.stderr


@pytest.mark.skipif(REAL_COG is None, reason="cog is not on PATH")
def test_real_cog_on_a_dirty_tree_fails(repo_1_13):
    release(repo_1_13, "1.13.1")
    pixi = repo_1_13 / "pixi.toml"
    pixi.write_text(f"{pixi.read_text()}\n")
    result = check(repo_1_13, cog="real")
    assert result.returncode == 1
    assert "cog bump --dry-run --auto failed: Error: " in result.stderr
    assert len(result.stderr.splitlines()) == 1


def test_release_commit_fails_when_cog_fails(repo):
    release(repo, "1.1.0")
    result = check(repo, next_version="")
    assert result.returncode == 1
    assert result.stderr.splitlines() == drift(
        ["cog bump --dry-run --auto failed: Error: bump is not allowed"]
    )


def test_release_commit_fails_when_cog_prints_no_version(repo):
    release(repo, "1.1.0")
    result = check(repo, next_version="nothing")
    assert result.returncode == 1
    assert result.stderr.splitlines() == drift(
        ["cog bump --dry-run --auto printed 'nothing', not a version"]
    )


@pytest.mark.parametrize("next_version", ["1.1.0", "v1.1.0"])
def test_release_commit_passes_before_its_tag(repo, next_version):
    release(repo, "1.1.0")
    result = check(repo, next_version=next_version)
    assert result.returncode == 0, result.stderr
    assert "version 1.1.0" in result.stdout


def test_release_commit_must_match_cog(repo):
    release(repo, "1.1.0")
    result = check(repo, next_version="1.0.1")
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "version drift: CHANGELOG.rst heading is 1.1.0, expected 1.0.1"
    ]


@pytest.mark.parametrize("version", ["0.9.0", "1.0.0"])
def test_release_commit_must_be_newer_than_the_tag(repo, version):
    release(repo, version)
    result = check(repo, next_version="v1.0.1")
    assert result.returncode == 1
    assert result.stderr.splitlines() == drift(
        [
            f"release heading {version} is not newer than v1.0.0",
            f"CHANGELOG.rst heading is {version}, expected 1.0.1",
        ]
    )


def test_deleted_tag_fails_after_the_release_commit(repo):
    release(repo, "1.1.0")
    git(repo, "tag", "v1.1.0")
    git(repo, "commit", "-q", "--allow-empty", "-m", "docs: after the release")
    assert check(repo).returncode == 0

    git(repo, "tag", "-d", "v1.1.0")
    result = check(repo)
    assert result.returncode == 1
    assert "heading is 1.1.0, expected 1.0.0" in result.stderr


def test_no_reachable_tag_fails(repo):
    git(repo, "tag", "-d", "v1.0.0")
    git(repo, "commit", "-q", "--allow-empty", "-m", "docs: untagged")
    result = check(repo)
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "version drift: no v* tag is reachable from HEAD"
    ]


def test_pixi_version_fails(repo):
    (repo / "pixi.toml").write_text('[workspace]\nname = "pycrumbs"\nversion = "1.0.0"\n')
    result = check(repo)
    assert result.returncode == 1
    assert "pixi.toml sets workspace.version" in result.stderr


def test_explicit_tag_must_match_the_heading(repo):
    assert check(repo, "v1.0.0").returncode == 0
    result = check(repo, "v1.0.1")
    assert result.returncode == 1
    assert "heading is 1.0.0, expected 1.0.1" in result.stderr
