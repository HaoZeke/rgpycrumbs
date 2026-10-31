"""A gallery profile computes and renders the saddle of each reaction only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ase")
pytest.importorskip("chemparseplot")
pytest.importorskip("matplotlib")

from ase import Atoms
from ase.io import write
from click.testing import CliRunner


def _profile(tmp_path, layout, states):
    lines = [
        "[reactions]",
        'output = "out"',
        "density = false",
        f'layout = "{layout}"',
        "columns = 2",
        "",
    ]
    for case in ("a", "b", "c"):
        lines += [
            "[[reactions.cases]]",
            f'id = "{case}"',
            f'title = "case {case}"',
            'subtitle = "H transfer"',
            "charge = 0",
            "multiplicity = 1",
        ]
        for title, name in states:
            lines += [
                "[[reactions.cases.states]]",
                f'title = "{title}"',
                f'path = "{name}.con"',
            ]
        lines.append("")
    path = tmp_path / "profile.toml"
    path.write_text("\n".join(lines))
    return path


def _structures(tmp_path):
    cell = [20, 20, 20]
    for name, x in (("reactant", 0.95), ("saddle", 1.25), ("product", 1.9)):
        atoms = Atoms(
            "OHH",
            positions=[[10, 10, 10], [10 + x, 10, 10], [10, 10.95, 10]],
            cell=cell,
        )
        write(tmp_path / f"{name}.con", atoms, format="eon")


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import numpy as np

    binary = tmp_path / "xtb"
    binary.write_text("test executable")
    monkeypatch.setattr("rgpycrumbs.geom._xtb.shutil.which", lambda _: str(binary))
    calculations = []

    def run(argv, **kwargs):
        if "--version" in argv:
            return SimpleNamespace(stdout="xTB test", stderr="", returncode=0)
        out = Path(kwargs["cwd"])
        calculations.append(out)
        (out / "wbo").write_text("1 2 0.45\n1 3 0.95\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rgpycrumbs.geom._xtb.subprocess.run", run)
    rendered = []

    def render(xyz, output, **kwargs):
        rendered.append(Path(output))
        plt.imsave(output, np.ones((8, 8, 4)))

    monkeypatch.setattr("chemparseplot.plot.reaction.render_reaction_state", render)
    return calculations, rendered


def test_gallery_computes_and_renders_only_the_saddle(tmp_path, fake_tools):
    from rgpycrumbs.chemgp.plot_reactions import cli

    calculations, rendered = fake_tools
    _structures(tmp_path)
    profile = _profile(
        tmp_path,
        "gallery",
        [("Reactant", "reactant"), ("GPRidge saddle", "saddle"), ("Product", "product")],
    )
    result = CliRunner().invoke(cli, ["--config", str(profile), "--xtb", "xtb"])
    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    # xTB runs in <case>/<state index>: the saddle (state 1) of each case.
    assert sorted((p.parent.name, p.name) for p in calculations) == [
        ("a", "1"),
        ("b", "1"),
        ("c", "1"),
    ]
    assert len(rendered) == 3
    assert (out / "reaction_gallery.pdf").is_file()
    assert not (out / "reaction_geometries.pdf").exists()
    rows = (out / "bonds.csv").read_text().splitlines()
    # One saddle state per case; the endpoints chose the changed O-H bond.
    assert len(rows) == 1 + 3
    assert all(",GPRidge saddle,1,2,O1\N{EN DASH}H2," in row for row in rows[1:])
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["calculations"]) == 3


def test_gallery_rejects_a_four_state_case(tmp_path, fake_tools):
    from rgpycrumbs.chemgp.plot_reactions import cli

    _structures(tmp_path)
    profile = _profile(
        tmp_path,
        "gallery",
        [
            ("Reactant", "reactant"),
            ("Saddle", "saddle"),
            ("Product", "product"),
            ("Other", "saddle"),
        ],
    )
    result = CliRunner().invoke(cli, ["--config", str(profile)])
    assert result.exit_code != 0
    assert "reactant, saddle and product" in str(result.exception)


def test_unknown_layout_is_refused(tmp_path, fake_tools):
    from rgpycrumbs.chemgp.plot_reactions import cli

    _structures(tmp_path)
    profile = _profile(tmp_path, "mosaic", [("Reactant", "reactant")])
    result = CliRunner().invoke(cli, ["--config", str(profile)])
    assert result.exit_code != 0
    assert "Unknown reaction layout" in str(result.exception)
