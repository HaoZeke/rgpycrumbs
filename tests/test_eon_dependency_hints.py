from pathlib import Path

from rgpycrumbs._aux import _DEPENDENCY_MAP


def test_eon_readcon_hints_require_spec_three():
    assert _DEPENDENCY_MAP["readcon"] == "readcon>=0.14.7"

    root = Path(__file__).parents[1] / "rgpycrumbs" / "eon"
    scripts = sorted(root.glob("*.py"))
    assert scripts
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        if "readcon>=0.14." in text:
            assert "readcon>=0.14.7" in text, path
            assert "readcon>=0.14.5" not in text, path


def test_chemparseplot_lazy_dependencies_require_current_eon_surface():
    specs = [value for key, value in _DEPENDENCY_MAP.items() if "chemparseplot" in key]
    assert specs
    for spec in specs:
        assert "chemparseplot[neb,plot]>=1.9.17,<2" == spec
