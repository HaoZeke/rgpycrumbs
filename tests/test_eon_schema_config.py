# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""eon-schema config write/hydrate integration (no eon-akmc)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("eon_schema")


def test_write_eon_config_roundtrip(tmp_path: Path):
    from eon_schema.config import read_ini

    from rgpycrumbs.eon.helpers import write_eon_config

    settings = {
        "Main": {"job": "saddle_search", "random_seed": 42},
        "Potential": {"potential": "lj"},
    }
    out = write_eon_config(tmp_path / "config.ini", settings)
    assert out.is_file()
    got = read_ini(out)
    assert got["Main"]["job"] == "saddle_search"
    assert got["Main"]["random_seed"] == "42"
    assert got["Potential"]["potential"] == "lj"


def test_write_eon_config_dir_and_validate(tmp_path: Path):
    from rgpycrumbs.eon.helpers import write_eon_config

    write_eon_config(tmp_path, {"Main": {"job": "point"}})
    assert (tmp_path / "config.ini").is_file()

    with pytest.raises(ValueError, match="unknown INI keys"):
        write_eon_config(
            tmp_path / "bad.ini",
            {"Main": {"job": "point", "not_a_real_key": 1}},
            validate=True,
        )


def test_write_eon_config_allows_uncovered_pot_section(tmp_path: Path):
    from rgpycrumbs.eon.helpers import write_eon_config

    # SocketNWChemPot is not L0-covered; validate must not flag it.
    write_eon_config(
        tmp_path / "sock.ini",
        {
            "Main": {"job": "saddle_search"},
            "SocketNWChemPot": {"unix_socket_path": "/tmp/x"},
        },
        validate=True,
    )


def test_log_config_ini_via_eon_schema(tmp_path: Path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow

    from rgpycrumbs.eon._mlflow.log_params import log_config_ini

    conf = tmp_path / "config.ini"
    conf.write_text("[Main]\njob = minimization\n")

    logged: dict[str, object] = {}

    def fake_log_param(k, v):
        logged[k] = v

    monkeypatch.setattr(mlflow, "log_param", fake_log_param)
    monkeypatch.setattr(mlflow, "set_tag", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **k: None)

    log_config_ini(conf, w_artifact=False, track_overrides=True)
    assert logged.get("Main/job") == "minimization"
    assert "Overrides/Main/job" in logged
    # hydrated defaults present
    assert any(k.startswith("Main/") for k in logged)
