import json

import numpy as np
import pytest
from click.testing import CliRunner


@pytest.mark.interpolation
def test_cubic_curvature_on_uneven_mass_weighted_path(tmp_path):
    from rgpycrumbs.eon.image_curvature_holdout import main

    masses = [1.0, 4.0]
    steps = np.zeros((5, 6))
    steps[::2, 0] = 1.0
    steps[1::2, 3] = 1.0
    positions = np.vstack((np.zeros(6), steps.cumsum(axis=0)))
    # Alternating unit displacements of the light and heavy atoms have lengths 1 and 2.
    arc = np.array([0, 1, 3, 4, 6, 7])
    blocks = (1 + arc**3)[:, None, None] * np.eye(6)
    source, seed, fitted = (tmp_path / name for name in ("blocks", "seed", "fitted"))
    source.write_text(
        json.dumps(
            {
                "positions_angstrom": positions.tolist(),
                "hessians_eV_per_angstrom2": blocks.tolist(),
            }
        )
    )
    seed.write_text(json.dumps({"masses": masses}))
    result = CliRunner().invoke(
        main,
        [
            str(source),
            str(seed),
            str(fitted),
            "--keep",
            "0",
            "--keep",
            "2",
            "--keep",
            "3",
            "--keep",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert [r["image"] for r in report["holdouts"]] == [1, 4]
    np.testing.assert_array_equal(report["knots_angstrom_sqrt_amu"], arc)
    np.testing.assert_allclose(
        json.loads(fitted.read_text())["hessians_eV_per_angstrom2"], blocks, atol=1e-12
    )
