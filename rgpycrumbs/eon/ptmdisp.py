#!/usr/bin/env python

# -----------------------------------------------------------------------------
# WARNING SUPPRESSION
# -----------------------------------------------------------------------------
import warnings

# Suppress the "OVITO...PyPI" UserWarning about installation environment.
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

# Suppress the "Please use atoms.calc" FutureWarning coming from ovito's ase import.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*Please use atoms.calc.*"
)
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import ase.io as aseio
from ovito.io import import_file
from ovito.io.ase import ase_to_ovito
from ovito.modifiers import PolyhedralTemplateMatchingModifier, SelectTypeModifier

from ovito.io import import_file, export_file
from ovito.pipeline import StaticSource, Pipeline
from ovito.modifiers import (
    SelectTypeModifier,
    PolyhedralTemplateMatchingModifier,
)


def get_disp_from(fname):
    atms = aseio.read(fname)
    atmdatc = ase_to_ovito(atms)
    pipeline = Pipeline(source=StaticSource(data=atmdatc))
    ptm = PolyhedralTemplateMatchingModifier()
    pipeline.modifiers.append(ptm)
    select_fcc = SelectTypeModifier(
        operate_on="particles",
        property="Structure Type",
        types={PolyhedralTemplateMatchingModifier.Type.FCC},
    )
    pipeline.modifiers.append(select_fcc)
    data = pipeline.compute()
    # The selection array marks selected (FCC) atoms with 1 and others with 0.
    non_fcc_indices = np.where(data.particles.selection.array == 0)[0]
    # 0-based indices into a single comma-separated string.
    # Example: [10, 23, 45] -> "10,23,45"
    return ",".join(map(str, non_fcc_indices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PTM FCC flipper",
        description="Selects non FCC atoms for use with EON",
    )
    parser.add_argument("filename")
    args = parser.parse_args()
    displace_list_str = get_disp_from(args.filename)
    print(displace_list_str)
