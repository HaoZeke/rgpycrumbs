"""Compare physical LI-NEB results and preserve both cache accounting records."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "numpy", "readcon"]
# ///

import configparser
import hashlib
import json
from pathlib import Path

import click
import numpy as np
import readcon


def differences(left, right):
    a, b = (np.asarray(value, dtype=float) for value in (left, right))
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} and {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("nonfinite numerical result")
    delta = np.abs(a - b)
    index = np.unravel_index(delta.argmax(), delta.shape) if delta.size else ()
    return {
        "shape": list(a.shape),
        "max_abs": float(delta.max(initial=0)),
        "index": list(map(int, index)),
    }


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def compare(cold, warm, require_absolute, rate_inputs=()):
    runs = (cold, warm)
    fields, inputs, counts, trajectories = {}, {}, {}, {}

    def record(name, a, b, frame=None):
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError(f"{name}: metadata keys differ")
            for k in a:
                record(f"{name}.{k}", a[k], b[k], frame)
        elif isinstance(a, str) or isinstance(a, bool) or a is None:
            if type(a) is not type(b) or a != b:
                raise ValueError(f"{name}: {a!r} differs from {b!r}")
        else:
            info = differences(a, b)
            if name not in fields or info["max_abs"] > fields[name]["max_abs"]:
                fields[name] = {**info, "frame": frame}

    configs = []
    for run in runs:
        config = configparser.ConfigParser(interpolation=None)
        with (run / "config.ini").open() as stream:
            config.read_file(stream)
        configs.append(config)
    inputs["config.ini"] = [digest(run / "config.ini") for run in runs]
    for config in configs:
        config.remove_option("Lineb", "result_cache")
    if {s: dict(configs[0][s]) for s in configs[0]} != {
        s: dict(configs[1][s]) for s in configs[1]
    }:
        raise ValueError("physical configuration differs")
    seed = [digest(run / cfg["Lineb"]["initial_path"]) for run, cfg in zip(runs, configs)]
    if seed[0] != seed[1]:
        raise ValueError("initial path differs")
    inputs["initial_path"] = seed[0]
    inputs["rate_inputs"] = [{"file": p.name, "sha256": digest(p)} for p in rate_inputs]

    def available(filename, required):
        exists = [(r / filename).is_file() for r in runs]
        if exists[0] != exists[1] or (required and not all(exists)):
            raise ValueError(f"incomplete pair for {filename}")
        return exists[0]

    source_counts = None
    if available("source-level.jsonl", require_absolute):
        levels = [digest(run / "source-level.jsonl") for run in runs]
        if levels[0] != levels[1]:
            raise ValueError("frozen source level differs")
        inputs["source-level.jsonl"] = levels[0]
        source = json.loads((cold / "source-level.jsonl").read_text())
        source_counts = {
            "band_calls": source["calls"],
            "evaluation_counts": source["evaluation_counts"],
        }

    for filename in ("lineb.con", "lineb-orbit.con"):
        if not available(filename, require_absolute or filename == "lineb.con"):
            continue
        frames = [readcon.read_con(str(r / filename)) for r in runs]
        if not frames[0] or len(frames[0]) != len(frames[1]):
            raise ValueError(f"{filename}: frame count differs or is zero")

        def identity(frame):
            return [
                {
                    "id": atom.atom_id,
                    "symbol": atom.symbol,
                    "mass_amu": atom.mass,
                    "fixed": list(atom.fixed),
                }
                for atom in frame.atoms
            ]

        signature = identity(frames[0][0])
        trajectories[filename] = {"frames": len(frames[0]), "atoms": signature}
        for i, (a, b) in enumerate(zip(*frames)):
            if identity(a) != signature or identity(b) != signature:
                raise ValueError(
                    f"{filename} frame {i}: atom IDs, species, masses or fixed flags differ"
                )
            for name in ("xyz", "cell", "angles"):
                record(f"{filename}.{name}", getattr(a, name), getattr(b, name), i)
            meta = [dict(f.metadata) for f in (a, b)]
            for tag, m in zip(("cold", "warm"), meta):
                counter = m.pop("evaluation_counts")
                if tag in counts and counter != counts[tag]["evaluation_counts"]:
                    raise ValueError(f"{filename}: inconsistent {tag} evaluation counts")
                counts[tag] = {"evaluation_counts": counter}
                if filename == "lineb-orbit.con":
                    orbit = m["lineb_orbit"]
                    ids = orbit["mobile_atom_ids"]
                    if len(set(ids)) != len(ids) or np.shape(
                        orbit["gradients_eV_per_angstrom"]
                    ) != (len(ids), 3):
                        raise ValueError("invalid orbit gradient shape or mobile IDs")
            record(f"{filename}.metadata", *meta, i)
    if available("native.json", require_absolute):
        native = [json.loads((r / "native.json").read_text()) for r in runs]
        for tag, n in zip(("cold", "warm"), native):
            if n["evaluation_counts"] != counts[tag]["evaluation_counts"]:
                raise ValueError(f"{tag}: native summary and CON counter mismatch")
            for key in ("physical_launches", "scf_convergence_reports"):
                counts[tag][key] = n[key]
        if native[0]["source"] != native[1]["source"]:
            raise ValueError("native input hashes differ")
        inputs["native_source"] = native[0]["source"]
        if inputs["native_source"].get("level") != inputs.get("source-level.jsonl"):
            raise ValueError("native source hash differs from the frozen level")
        for tag in ("cold", "warm"):
            ledger = counts[tag]
            if (
                ledger["physical_launches"]
                != ledger["evaluation_counts"]["rgpot_results"]["computed"]
            ):
                raise ValueError(f"{tag}: physical launches differ from computed results")
        if counts["warm"]["physical_launches"]:
            raise ValueError("warm replay launched physical evaluations")
        for key in (
            "source_level",
            "source_level_counts",
            "reference_energy_eV",
            "saddle_energy_eV",
        ):
            record(f"native.{key}", native[0][key], native[1][key])
    if available("absolute_rate.json", require_absolute):
        rates = [json.loads((r / "absolute_rate.json").read_text()) for r in runs]
        if not all("log10_rate_per_s" in rate for rate in rates):
            raise ValueError("absolute rate is missing")
        record("absolute_rate", *rates)
    for key in ("surface_calls", "potential_calls"):
        if (
            counts["cold"]["evaluation_counts"][key]
            != counts["warm"]["evaluation_counts"][key]
        ):
            raise ValueError(f"{key}: number of requested evaluations differs")
    return {
        "identical_physics": all(d["max_abs"] == 0 for d in fields.values()),
        "input_sha256": inputs,
        "trajectories": trajectories,
        "source_level_counts": source_counts,
        "counts": counts,
        "differences": fields,
    }


@click.command()
@click.argument("cold", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("warm", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--require-absolute",
    is_flag=True,
    help="Require native counters, measured orbit gradients and molecular rate.",
)
@click.option(
    "--rate-input",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Record a shared input used for absolute-rate normalization.",
)
def main(cold, warm, require_absolute, rate_input):
    """Compare recorded cold and warm eOn jobs with identical physical inputs."""
    report = compare(cold, warm, require_absolute, rate_input)
    click.echo(json.dumps(report, indent=2, allow_nan=False))
    if not report["identical_physics"]:
        raise click.ClickException("physical outputs differ")


if __name__ == "__main__":
    main()
