# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Coverage for suite DRY batch: suite_pins, surface_fit_config, basetypes, chemgp."""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from rgpycrumbs.api import suite_pins
from rgpycrumbs.basetypes import DimerOpt, MolGeom, SaddleMeasure, SpinID, nebiter, nebpath
from rgpycrumbs.eon.plot_config import (
    click_nondefault_overrides,
    extract_config_layers,
    load_plot_config,
    merge_plot_settings,
    resolve_from_click,
    surface_fit_config,
)

pytestmark = pytest.mark.pure


def test_suite_pins():
    assert isinstance(suite_pins(), dict)


def test_suite_pins_soft_fail(monkeypatch):
    import rgpycrumbs.api as api

    def boom(*_a, **_k):
        raise RuntimeError("cfg")

    monkeypatch.setattr(api, "load_config", boom)
    pins = api.suite_pins()
    assert isinstance(pins, dict)


def test_surface_fit_config_with_defaults():
    s = merge_plot_settings(
        "min", config_data={"shared": {"auto_thin": True, "max_surface_points": 8}}
    )
    cfg = surface_fit_config(s)
    assert cfg.auto_thin is True
    assert cfg.max_surface_points == 8


def test_surface_fit_config_fallback(monkeypatch):
    import builtins

    import rgpycrumbs.eon.plot_config as pc

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("chemparseplot"):
            raise ImportError("no cpp")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        pc,
        "SurfaceFitConfig",
        None,
        raising=False,
    )
    cfg = surface_fit_config({"auto_thin": False, "max_surface_points": 16})
    assert cfg.max_surface_points == 16
    assert cfg.auto_thin is False


def test_basetypes_reexport():
    p = nebpath(0.0, 0.0, 0.0)
    assert nebiter(1, p).iteration == 1
    assert DimerOpt().saddle == "dimer"
    assert SpinID(1, "s").spin == "s"
    g = MolGeom(np.zeros((1, 3)), 0.0, np.zeros((1, 3)))
    assert g.energy == 0.0
    assert SaddleMeasure().success is False


def test_basetypes_fallback(monkeypatch):
    sys.modules.pop("rgpycrumbs.basetypes", None)

    import builtins

    real = builtins.__import__

    def fake(name, globs=None, locs=None, fromlist=(), level=0):
        if name.startswith("chemparseplot"):
            raise ImportError("hidden")
        return real(name, globs, locs, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake)
    import rgpycrumbs.basetypes as bt

    p = bt.nebpath(1.0, 2.0, 3.0)
    assert bt.nebiter(0, p).nebpath.arc_dist == 2.0
    sys.modules.pop("rgpycrumbs.basetypes", None)
    importlib.invalidate_caches()


def test_chemgp_deprecation_warning():
    sys.modules.pop("rgpycrumbs.chemgp", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import rgpycrumbs.chemgp as _cg  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"chemgp deps missing: {exc}")
        deps = [x for x in caught if issubclass(x.category, DeprecationWarning)]
        assert deps, "expected DeprecationWarning on chemgp import"


def test_load_plot_config_and_layers(tmp_path):
    p = tmp_path / "plot.toml"
    p.write_text(
        '[shared]\nauto_thin = true\nmax_surface_points = 10\nenergy_unit = "eV"\n'
        '[min]\nprefix = "m"\n'
        "extra_flat = 1\n"
    )
    data = load_plot_config(p)
    shared, _cmd = extract_config_layers(data, "min")
    assert shared["auto_thin"] is True
    settings = merge_plot_settings("min", config_path=p, cli_overrides={"dpi": 150})
    assert settings["dpi"] == 150
    assert settings["auto_thin"] is True


def test_merge_unknown_command():
    with pytest.raises(ValueError, match="Unknown"):
        merge_plot_settings("nope")


def test_merge_passthrough_and_job_dir(tmp_path):
    settings = merge_plot_settings(
        "min",
        config_data={},
        passthrough={"custom": 1},
        cli_overrides={"job_dir": str(tmp_path)},
    )
    assert settings["custom"] == 1
    assert settings["job_dir"][0] == Path(tmp_path)


def test_resolve_from_click_like():
    class Ctx:
        def get_parameter_source(self, name):
            from click.core import ParameterSource

            if name == "dpi":
                return ParameterSource.COMMANDLINE
            return ParameterSource.DEFAULT

    settings = resolve_from_click("neb", Ctx(), config=None, dpi=300, title="t")
    assert settings["dpi"] == 300


def test_plot_config_coercions(tmp_path):
    from rgpycrumbs.eon.plot_config import _coerce_value

    assert _coerce_value("figsize", [1, 2]) == (1.0, 2.0)
    assert _coerce_value("label", "a") == ("a",)
    assert _coerce_value("label", ["a", "b"]) == ("a", "b")
    assert _coerce_value("job_dir", str(tmp_path)) == [Path(tmp_path)]
    assert _coerce_value("job_dir", [str(tmp_path)])[0] == Path(tmp_path)
    assert _coerce_value("con_file", "x.con") == Path("x.con")


def test_merge_label_and_list_job_dir(tmp_path):
    s = merge_plot_settings(
        "min",
        config_data={"min": {"label": ["a"], "job_dir": [str(tmp_path)]}},
    )
    assert s["label"] == ("a",)
    assert isinstance(s["job_dir"], tuple)


def test_click_nondefault():
    from click.core import ParameterSource

    class Ctx:
        def get_parameter_source(self, name):
            if name == "dpi":
                return ParameterSource.COMMANDLINE
            if name == "config":
                return ParameterSource.COMMANDLINE
            return ParameterSource.DEFAULT

    ov = click_nondefault_overrides(Ctx(), {"dpi": 99, "config": "x", "title": "t"})
    assert "dpi" in ov
    assert "config" not in ov


def test_load_plot_config_non_table(monkeypatch, tmp_path):
    import rgpycrumbs.eon.plot_config as pc

    p = tmp_path / "x.toml"
    p.write_text("1")
    monkeypatch.setattr(pc.tomllib, "loads", lambda _b: 123)
    with pytest.raises(ValueError, match="table"):
        pc.load_plot_config(p)


def test_coerce_none_and_job_dir_list_of_paths():
    from rgpycrumbs.eon.plot_config import _coerce_value

    assert _coerce_value("dpi", None) is None
    assert _coerce_value("job_dir", [Path("a"), Path("b")])[0] == Path("a")
    assert _coerce_value("label", ("x",)) == ("x",)


def test_extract_nested_dict_skipped():
    shared, cmd = extract_config_layers(
        {"shared": {"dpi": 100}, "min": {"prefix": "p"}, "nested": {"a": 1}, "flat": 2},
        "min",
    )
    assert "nested" not in shared
    assert cmd.get("flat") == 2


def test_merge_cli_none_skip_and_job_dir_str_label_str(tmp_path):
    s = merge_plot_settings(
        "min",
        config_data={},
        cli_overrides={
            "weird": None,
            "job_dir": str(tmp_path),
            "label": "lab",
            "dpi": None,
        },
    )
    assert "weird" not in s or s.get("weird") is None
    assert s["job_dir"][0] == Path(tmp_path)
    assert s["label"] == ("lab",)


def test_merge_job_dir_tuple_and_label_list(tmp_path):
    s = merge_plot_settings(
        "min",
        config_data={
            "min": {
                "job_dir": (str(tmp_path),),
                "label": ["x", "y"],
            }
        },
    )
    assert isinstance(s["job_dir"], tuple)
    assert s["label"] == ("x", "y")


def test_merge_job_dir_path_and_list_normalize(tmp_path):
    s = merge_plot_settings(
        "min",
        config_data={},
        cli_overrides={"job_dir": tmp_path},
    )
    assert s["job_dir"] == (tmp_path,)

    s2 = merge_plot_settings(
        "min",
        config_data={},
        passthrough={"job_dir": [tmp_path / "a", tmp_path / "b"]},
    )
    assert len(s2["job_dir"]) == 2


def test_coerce_unknown_key_passthrough():
    from rgpycrumbs.eon.plot_config import _coerce_value

    assert _coerce_value("other", 1) == 1


def test_extract_shared_key_at_top_level():
    shared, cmd = extract_config_layers({"auto_thin": True, "prefix": "z"}, "min")
    assert shared["auto_thin"] is True
    assert cmd["prefix"] == "z"


def test_merge_label_already_str_from_settings(tmp_path):
    s = merge_plot_settings(
        "min",
        config_data={"min": {"label": "solo", "job_dir": [str(tmp_path)]}},
        cli_overrides={"label": ["a", "b"]},
    )
    assert s["label"] == ("a", "b")


def test_merge_job_dir_list_and_label_list_via_settings():
    s = merge_plot_settings(
        "min",
        config_data={},
        cli_overrides={"job_dir": ["a", "b"], "label": ["L1"]},
    )
    assert s["job_dir"] == (Path("a"), Path("b"))
    assert s["label"] == ("L1",)


def test_merge_job_dir_tuple_normalize():
    s = merge_plot_settings(
        "min",
        config_data={},
        cli_overrides={"job_dir": (Path("x"), Path("y"))},
    )
    assert s["job_dir"] == (Path("x"), Path("y"))


def test_merge_passthrough_raw_job_dir_and_label_shapes(tmp_path):
    s = merge_plot_settings(
        "min",
        config_data={},
        passthrough={
            "job_dir": str(tmp_path),
            "label": "solo",
        },
    )
    assert s["job_dir"] == (Path(tmp_path),)
    assert s["label"] == ("solo",)

    s2 = merge_plot_settings(
        "min",
        config_data={},
        passthrough={
            "job_dir": [str(tmp_path / "a")],
            "label": ["a", "b"],
        },
    )
    assert s2["job_dir"] == (Path(tmp_path / "a"),)
    assert s2["label"] == ("a", "b")

    s3 = merge_plot_settings(
        "min",
        config_data={},
        passthrough={
            "job_dir": (tmp_path / "t",),
        },
    )
    assert s3["job_dir"] == (tmp_path / "t",)


def test_merge_label_non_sequence_left_alone():
    s = merge_plot_settings(
        "min",
        config_data={},
        passthrough={"label": 42},
    )
    assert s["label"] == 42
