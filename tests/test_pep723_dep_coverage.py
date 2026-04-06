# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Test that PEP 723 inline script deps cover every import in dispatched scripts.

When rgpycrumbs dispatches a script via `uv run`, only the dependencies
declared in the `# /// script` metadata block are available. Any import
not covered by those declarations (or the stdlib/rgpycrumbs itself) will
fail at runtime. This test catches such gaps statically.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pure

PACKAGE_ROOT = Path(__file__).parent.parent / "rgpycrumbs"

# Well-known top-level import -> PyPI package mapping for packages where
# the import name differs from the pip name.
_IMPORT_TO_PKG: dict[str, str] = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "Bio": "biopython",
    "gi": "pygobject",
    "Crypto": "pycryptodome",
    "serial": "pyserial",
    "usb": "pyusb",
    "wx": "wxpython",
    "pkg_resources": "setuptools",
    "dateutil": "python-dateutil",
    "google": "google-api-python-client",
    "IPython": "ipython",
    "skimage": "scikit-image",
    "mpl_toolkits": "matplotlib",
    "adjustText": "adjusttext",
    "cmcrameri": "cmcrameri",
    "resvg": "resvg-py",
}

# Packages that are part of rgpycrumbs itself or its internal imports.
_INTERNAL = {"rgpycrumbs", "chemparseplot"}

# Modules that are conda/pixi-only (no pip package) -- these cannot
# appear in PEP 723 deps and are imported conditionally.
_CONDA_ONLY = {"ira_mod", "tblite", "ovito", "eon"}

# Standard library module names (populated from sys.stdlib_module_names
# when available, otherwise a conservative fallback).
_STDLIB: set[str] = getattr(sys, "stdlib_module_names", set()) | {
    # Always include these in case of older Python
    "sys",
    "os",
    "pathlib",
    "logging",
    "re",
    "json",
    "math",
    "functools",
    "itertools",
    "collections",
    "typing",
    "dataclasses",
    "abc",
    "io",
    "warnings",
    "contextlib",
    "copy",
    "enum",
    "textwrap",
    "shutil",
    "subprocess",
    "tempfile",
    "argparse",
    "unittest",
    "hashlib",
    "struct",
    "operator",
    "importlib",
    "inspect",
    "string",
    "glob",
    "fnmatch",
    "time",
    "datetime",
    "calendar",
    "random",
    "statistics",
    "decimal",
    "fractions",
    "numbers",
    "array",
    "queue",
    "threading",
    "multiprocessing",
    "concurrent",
    "socket",
    "http",
    "urllib",
    "email",
    "html",
    "xml",
    "csv",
    "configparser",
    "tomllib",
    "pprint",
    "traceback",
    "pickle",
    "shelve",
    "sqlite3",
    "gzip",
    "zipfile",
    "tarfile",
    "lzma",
    "bz2",
    "site",
    "sysconfig",
    "platform",
    "signal",
    "ctypes",
    "weakref",
    "types",
    "gc",
    "dis",
    "ast",
    "token",
    "tokenize",
    "pdb",
    "profile",
    "cProfile",
    "timeit",
    "resource",
    "errno",
    "select",
    "selectors",
    "mmap",
    "codecs",
    "locale",
    "gettext",
    "unicodedata",
    "stringprep",
    "readline",
    "rlcompleter",
    "textwrap",
    "difflib",
    "pydoc",
    "doctest",
    "secrets",
    "uuid",
    "base64",
    "binascii",
    "hmac",
    "ssl",
    "ftplib",
    "poplib",
    "imaplib",
    "smtplib",
    "xmlrpc",
    "asyncio",
    "__future__",
}


def _parse_pep723_deps(source: str) -> set[str]:
    """Extract normalized package names from a PEP 723 script block."""
    # Match the /// script ... /// block
    match = re.search(
        r"^# /// script\s*\n((?:#[^\n]*\n)*?)# ///",
        source,
        re.MULTILINE,
    )
    if not match:
        return set()
    block = match.group(1)
    # Strip leading '#' from each line, join, then extract from the
    # dependencies list.  This handles both one-dep-per-line and
    # multiple-deps-on-one-line formats.
    stripped = "\n".join(line.lstrip("#").strip() for line in block.splitlines())
    # Find the dependencies = [...] section
    dep_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", stripped, re.DOTALL)
    if not dep_match:
        return set()
    deps_block = dep_match.group(1)
    deps = set()
    for m in re.finditer(r'"([^"]+)"', deps_block):
        raw = m.group(1)
        # Normalize: strip version specs, extras, whitespace
        pkg = re.split(r"[>=<!\[;~]", raw)[0].strip().lower()
        deps.add(pkg)
    return deps


def _extract_imports(source: str) -> set[str]:
    """Extract all top-level package names from import statements."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                modules.add(node.module.split(".")[0])
    return modules


def _find_pep723_scripts() -> list[Path]:
    """Find all .py files under PACKAGE_ROOT with PEP 723 script blocks."""
    scripts = []
    for py_file in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        content = py_file.read_text()
        if "# /// script" in content:
            scripts.append(py_file)
    return sorted(scripts)


def _normalize_import_to_pkg(imp: str) -> str:
    """Map an import name to the expected PyPI package name."""
    if imp in _IMPORT_TO_PKG:
        return _IMPORT_TO_PKG[imp].lower()
    return imp.lower().replace("_", "-")


class TestPEP723DepCoverage:
    """Verify inline script deps cover all imports in dispatched scripts."""

    @pytest.fixture(scope="class")
    def scripts(self) -> list[tuple[Path, str]]:
        """Collect all PEP 723 scripts and their source."""
        result = []
        for path in _find_pep723_scripts():
            result.append((path, path.read_text()))
        return result

    def test_found_scripts(self, scripts) -> None:
        """Sanity check: we should find at least a few PEP 723 scripts."""
        assert len(scripts) >= 5, (
            f"Expected >=5 PEP 723 scripts, found {len(scripts)}. "
            f"Check PACKAGE_ROOT: {PACKAGE_ROOT}"
        )

    def test_all_imports_covered(self, scripts) -> None:
        """Every import in a PEP 723 script must be resolvable from the
        declared inline deps, stdlib, or internal packages."""
        failures = []
        for path, source in scripts:
            declared_deps = _parse_pep723_deps(source)
            imports = _extract_imports(source)

            for imp in imports:
                # Skip stdlib
                if imp in _STDLIB:
                    continue
                # Skip internal packages
                if imp in _INTERNAL:
                    continue
                # Skip conda-only
                if imp in _CONDA_ONLY:
                    continue
                # Check if import maps to a declared dep
                expected_pkg = _normalize_import_to_pkg(imp)
                if expected_pkg not in declared_deps:
                    # Also check the raw import name (for cases like cmcrameri)
                    if imp.lower() not in declared_deps:
                        rel = path.relative_to(PACKAGE_ROOT.parent)
                        failures.append(
                            f"  {rel}: `import {imp}` -> "
                            f"expected '{expected_pkg}' in deps, "
                            f"found: {sorted(declared_deps)}"
                        )

        assert not failures, (
            "PEP 723 scripts have imports not covered by inline deps:\n"
            + "\n".join(failures)
            + "\n\nFix: add the missing package to the # /// script "
            "dependencies block."
        )
