The fake-dependency tests in ``tests/test_coverage_boost.py`` unbind each ``rgpycrumbs`` submodule they drop from ``sys.modules``.
Test files that run later import those submodules against the real dependencies.
