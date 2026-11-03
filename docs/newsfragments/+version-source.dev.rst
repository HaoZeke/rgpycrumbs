A CI step, ``scripts/check_version.py``, fails when ``pixi.toml`` sets a version.
It also fails when the newest ``CHANGELOG.rst`` heading is not the git tag.
A release commit before its tag must name a version newer than that tag.
It must also equal the next version from ``cog bump --dry-run --auto``.
The check fails when cog is not on ``PATH`` or prints no version.
