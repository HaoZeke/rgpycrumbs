#!/bin/bash
# Pre-commit hook: regenerate README.md from readme_src.org
# Requires emacs; silently skips if unavailable.
if ! command -v emacs &>/dev/null; then
    exit 0
fi
bash scripts/org_to_md.sh readme_src.org README.md
git add README.md
