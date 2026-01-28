# Usage:
# https://atomistic-cookbook.org/examples/eon-pet-neb/eon-pet-neb.html

import configparser
from pathlib import Path
from typing import Any, Union


def write_eon_config(path: str | Path, settings: dict[str, dict[str, Any]]) -> None:
    """
    Writes a config.ini file for eOn using a dictionary structure.
    """
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity

    for section, options in settings.items():
        config[section] = {k: str(v) for k, v in options.items()}

    out_path = Path(path)
    if out_path.is_dir():
        out_path = out_path / "config.ini"

    with open(out_path, "w") as f:
        config.write(f)
    print(f"Wrote eOn config to '{out_path}'")
