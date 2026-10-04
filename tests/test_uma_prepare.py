# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""UMA AOTI key, cache and prepare path; no fairchem or torch.

A stand-in exporter writes a ``.pt2``-shaped zip whose embedded metadata
has the layout torch writes for rgpot's exporter.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from rgpycrumbs.uma import (
    AotiKey,
    ExportEnv,
    aoti_key,
    embedded_metadata,
    embedded_mismatches,
    entries,
    exact_counts,
    find_entry,
    hill_formula,
    lookup,
    minimal_spin,
    package_path,
    prepare_uma_aoti,
    probe_env,
    reduced_counts,
    remove,
    resolve_exporter,
    sidecar_path,
    validate_spin,
)
from rgpycrumbs.uma._cache import key_lock, stale_partials
from rgpycrumbs.uma._env import normalize_device

pytestmark = pytest.mark.pure

ENV = ExportEnv(device="cpu", target="x86_64|AVX512|", torch="2.13.0", fairchem="2.23.0")
HCN = [6, 7, 1]
ROOT = Path(__file__).resolve().parent.parent

FAKE_EXPORTER = textwrap.dedent(
    """
    import argparse, json, os, sys, time, zipfile
    from pathlib import Path

    p = argparse.ArgumentParser()
    for name in ("atoms", "charge", "spin", "task", "model", "device", "label", "out"):
        p.add_argument("--" + name)
    p.add_argument("--molecular-box", default="0")
    p.add_argument("--batch-max", default="0")
    a = p.parse_args()
    with open(os.environ["FAKE_COUNT"], "a") as fh:
        fh.write(a.label + "\\n")
    time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
    code = int(os.environ.get("FAKE_EXIT", "0"))
    if code:
        sys.exit(code)
    z = [int(v) for v in os.environ["FAKE_Z"].split(",")]
    systems = max(int(a.batch_max), 1)
    meta = {
        "cutoff": "6.0",
        "molecular_box": str(float(a.molecular_box)),
        "max_neighbors": "300",
        "task_name": a.task,
        "charge": a.charge,
        "spin": os.environ.get("FAKE_SPIN", a.spin),
        "z_set": str(sorted(set(z))),
        "label": a.label,
        "batch_max": a.batch_max,
        "pos_dtype": "float32",
        "shapes": str({"pos": [systems * len(z), 3]}),
        "AOTI_DEVICE_KEY": a.device.split(":")[0],
        "AOTI_PLATFORM": "linux",
        "AOTI_MACHINE": "x86_64",
        "AOTI_CPU_ISA": "AVX512",
        "AOTI_COMPUTE_CAPABILITY": "",
    }
    out = Path(a.out)
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(out.stem + "/data/aotinductor/model/abc.wrapper_metadata.json", json.dumps(meta))
        zf.writestr(out.stem + "/data/aotinductor/model/abc.wrapper.so", b"so")
    """
)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Stand-in exporter plus its call log; returns (exporter, count_file)."""
    exporter = tmp_path / "export_uma_aoti.py"
    exporter.write_text(FAKE_EXPORTER)
    count = tmp_path / "calls.txt"
    count.write_text("")
    monkeypatch.setenv("FAKE_COUNT", str(count))
    monkeypatch.setenv("FAKE_Z", ",".join(str(z) for z in HCN))
    for var in ("FAKE_SLEEP", "FAKE_EXIT", "FAKE_SPIN"):
        monkeypatch.delenv(var, raising=False)
    return exporter, count


def _calls(count: Path) -> int:
    return len(count.read_text().splitlines())


def _prepare(cache: Path, exporter: Path, **kw):
    kw.setdefault("atoms_path", exporter)
    return prepare_uma_aoti(HCN, cache_dir=cache, exporter=exporter, env=ENV, **kw)


class TestCounts:
    def test_reduced_is_model_side_only(self):
        c2h2 = reduced_counts([6, 6, 1, 1])
        c4h4 = reduced_counts([6, 6, 6, 6, 1, 1, 1, 1])
        assert c2h2 == ((1, 1), (6, 1))
        assert c2h2 == c4h4

    def test_exact_separates_acetylene_from_double(self):
        assert exact_counts([6, 6, 1, 1]) == ((1, 2), (6, 2))
        assert exact_counts([6, 6, 1, 1]) != exact_counts([6, 6, 6, 6, 1, 1, 1, 1])

    def test_hcn(self):
        assert exact_counts([6, 7, 1]) == ((1, 1), (6, 1), (7, 1))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            exact_counts([])

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            exact_counts([0, 1])

    @pytest.mark.parametrize(
        ("numbers", "formula"),
        [
            ([6, 7, 1], "CHN"),
            ([6, 6, 1, 1], "C2H2"),
            ([8, 1, 1], "H2O"),
            ([26, 8], "FeO"),
        ],
    )
    def test_hill_formula(self, numbers, formula):
        assert hill_formula(exact_counts(numbers)) == formula


class TestSpinParity:
    def test_even_electrons_default_singlet(self):
        assert minimal_spin([6, 7, 1]) == 1

    def test_odd_electrons_default_doublet(self):
        # cyclopropyl C3H5: 23 electrons
        assert minimal_spin([6, 6, 6, 1, 1, 1, 1, 1]) == 2

    def test_charge_flips_parity(self):
        assert minimal_spin([6, 6, 6, 1, 1, 1, 1, 1], charge=1) == 1

    def test_singlet_radical_rejected(self):
        with pytest.raises(ValueError, match="impossible for 23 electrons"):
            validate_spin([6, 6, 6, 1, 1, 1, 1, 1], spin=1)

    def test_multiplicity_above_electron_count_rejected(self):
        # H2: 2 electrons admit multiplicity 1 or 3, never 5
        validate_spin([1, 1], spin=3)
        with pytest.raises(ValueError, match="needs more than 2 electrons"):
            validate_spin([1, 1], spin=5)

    def test_key_derives_doublet_for_radical(self):
        assert aoti_key([6, 6, 6, 1, 1, 1, 1, 1], ENV).spin == 2

    def test_key_rejects_parity_mismatch(self):
        with pytest.raises(ValueError, match="impossible"):
            aoti_key([6, 7, 1], ENV, spin=2)


class TestKey:
    def test_stem_names_the_system_and_ends_in_the_digest(self):
        key = aoti_key(HCN, ENV)
        assert key.stem() == f"omol-CHN-q0-s1-uma-s-1p1-cpu-{key.digest()}"
        assert len(key.digest()) == 20

    def test_digest_is_stable(self):
        # Changing this value orphans every cached package: bump KEY_SCHEMA.
        assert aoti_key(HCN, ENV).digest() == "0a7cfb8cf7b60b95cc24"

    def test_round_trip(self):
        key = aoti_key([6, 6, 1, 1], ENV, charge=1, molecular_box=25, batch_max=8)
        assert AotiKey.from_dict(json.loads(json.dumps(key.as_dict()))) == key

    def test_every_field_separates_keys(self):
        base = aoti_key(HCN, ENV)
        variants = {
            "counts": ((1, 2), (6, 2), (7, 2)),
            "charge": 2,
            "spin": 3,
            "task": "omat",
            "model": "uma-m-1p1",
            "dtype": "float64",
            "molecular_box": "25",
            "batch_max": 8,
            "device": "cuda",
            "target": "x86_64|AVX2|",
            "torch": "2.10.0",
            "fairchem": "2.21.0",
            "schema": base.schema + 1,
        }
        assert set(variants) == {f.name for f in dataclasses.fields(AotiKey)}
        keys = [base] + [dataclasses.replace(base, **{k: v}) for k, v in variants.items()]
        assert len({k.digest() for k in keys}) == len(keys)
        assert len({k.stem() for k in keys}) == len(keys)

    def test_reduced_equal_compositions_do_not_collide(self):
        a = aoti_key([6, 6, 1, 1], ENV)
        b = aoti_key([6, 6, 6, 6, 1, 1, 1, 1], ENV)
        assert a.digest() != b.digest()

    def test_atom_order_does_not_matter(self):
        assert aoti_key([1, 6, 7], ENV) == aoti_key([7, 1, 6], ENV)

    def test_equivalent_spellings_share_a_key(self):
        assert aoti_key(HCN, ENV, molecular_box=25) == aoti_key(
            HCN, ENV, molecular_box=25.0
        )
        assert aoti_key(HCN, ENV, batch_max=1) == aoti_key(HCN, ENV, batch_max=0)

    def test_negative_options_rejected(self):
        with pytest.raises(ValueError, match="molecular_box"):
            aoti_key(HCN, ENV, molecular_box=-1)
        with pytest.raises(ValueError, match="batch_max"):
            aoti_key(HCN, ENV, batch_max=-2)


def _meta(key: AotiKey, **over) -> dict:
    meta = {
        "task_name": key.task,
        "charge": str(key.charge),
        "spin": str(key.spin),
        "z_set": str(key.z_set),
        "shapes": str({"pos": [key.natoms, 3]}),
        "pos_dtype": key.dtype,
        "molecular_box": "0.0",
        "batch_max": "0",
        "AOTI_DEVICE_KEY": "cpu",
        "AOTI_MACHINE": "x86_64",
        "AOTI_CPU_ISA": "AVX512",
        "AOTI_COMPUTE_CAPABILITY": "",
    }
    meta.update(over)
    return meta


class TestEmbedded:
    def test_matching_metadata_passes(self):
        key = aoti_key(HCN, ENV)
        assert embedded_mismatches(_meta(key), key) == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("spin", "3"),
            ("charge", "1"),
            ("task_name", "omat"),
            ("z_set", "[1, 6]"),
            ("shapes", "{'pos': [6, 3]}"),
            ("pos_dtype", "float64"),
            ("molecular_box", "25.0"),
            ("batch_max", "4"),
            ("AOTI_DEVICE_KEY", "cuda"),
            ("AOTI_CPU_ISA", "AVX2"),
        ],
    )
    def test_each_field_is_checked(self, field, value):
        key = aoti_key(HCN, ENV)
        bad = embedded_mismatches(_meta(key, **{field: value}), key)
        assert len(bad) == 1

    def test_exporter_provenance_is_checked_when_present(self):
        key = aoti_key(HCN, ENV)
        counts = json.dumps({str(z): n for z, n in key.counts})
        full = _meta(
            key,
            natoms=str(key.natoms),
            counts=counts,
            model=key.model,
            torch_version=key.torch,
            fairchem_version=key.fairchem,
        )
        assert embedded_mismatches(full, key) == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("natoms", "4"),
            ("counts", json.dumps({"1": 2, "6": 1, "7": 1})),
            ("model", "uma-m-1p1"),
            ("torch_version", "0.0.1"),
            ("fairchem_version", "0.0.1"),
        ],
    )
    def test_each_exporter_field_is_checked(self, field, value):
        key = aoti_key(HCN, ENV)
        bad = embedded_mismatches(_meta(key, **{field: value}), key)
        assert len(bad) == 1, bad
        assert bad[0].startswith("embedded natoms" if field == "natoms" else field)

    def test_counts_separate_compositions_with_one_element_set(self):
        # C3H3 and C2H4 share z_set [1, 6] and six atoms, so the element
        # set and the traced atom count pass; only the counts, which
        # UmaPot checks on every call, tell them apart.
        key = aoti_key([6, 6, 6, 1, 1, 1], ENV)
        meta = _meta(key, counts=json.dumps({"1": 4, "6": 2}))
        bad = embedded_mismatches(meta, key)
        assert len(bad) == 1 and bad[0].startswith("counts"), bad

    def test_band_package_counts_atoms_per_system(self):
        key = aoti_key(HCN, ENV, batch_max=4)
        meta = _meta(key, batch_max="4", shapes=str({"pos": [12, 3]}))
        assert embedded_mismatches(meta, key) == []

    def test_reads_the_torch_layout(self, tmp_path: Path):
        pt2 = tmp_path / "x.pt2"
        with zipfile.ZipFile(pt2, "w") as zf:
            zf.writestr(
                "x/data/aotinductor/model/k.wrapper_metadata.json", '{"spin": "1"}'
            )
            zf.writestr("x/archive_format", "pt2")
        assert embedded_metadata(pt2) == {"spin": "1"}

    def test_package_without_metadata_raises(self, tmp_path: Path):
        pt2 = tmp_path / "x.pt2"
        with zipfile.ZipFile(pt2, "w") as zf:
            zf.writestr("x/archive_format", "pt2")
        with pytest.raises(ValueError, match="no AOTI metadata"):
            embedded_metadata(pt2)


class TestLookup:
    def test_package_without_sidecar_misses(self, tmp_path: Path):
        key = aoti_key(HCN, ENV)
        package_path(tmp_path, key).write_bytes(b"pt2")
        assert lookup(tmp_path, key) is None

    def test_sidecar_of_another_key_misses(self, tmp_path: Path):
        key = aoti_key(HCN, ENV)
        other = dataclasses.replace(key, spin=3)
        pt2 = package_path(tmp_path, key)
        pt2.write_bytes(b"pt2")
        sidecar_path(pt2).write_text(json.dumps({"key": other.as_dict()}))
        assert lookup(tmp_path, key) is None
        sidecar_path(pt2).write_text(json.dumps({"key": key.as_dict()}))
        assert lookup(tmp_path, key) == pt2

    def test_sidecar_name_follows_rgpot(self, tmp_path: Path):
        assert sidecar_path(tmp_path / "m.pt2").name == "m.pt2.json"


class TestPrepare:
    def test_miss_compiles_then_hits(self, tmp_path: Path, fake):
        exporter, count = fake
        cache = tmp_path / "cache"
        first = _prepare(cache, exporter)
        assert not first.hit
        assert first.compile_seconds is not None
        assert first.path.is_file()
        side = json.loads(sidecar_path(first.path).read_text())
        assert side["key"] == first.key.as_dict()
        assert side["embedded"]["label"] == first.key.stem()
        second = _prepare(cache, exporter)
        assert second.hit
        assert second.path == first.path
        assert _calls(count) == 1
        assert not list(cache.glob(".partial-*"))

    def test_sidecar_is_as_readable_as_the_package(self, tmp_path: Path, fake):
        exporter, _count = fake
        old = os.umask(0o022)
        try:
            res = _prepare(tmp_path, exporter)
        finally:
            os.umask(old)
        assert sidecar_path(res.path).stat().st_mode & 0o777 == 0o644

    def test_hit_needs_no_structure(self, tmp_path: Path, fake):
        exporter, _count = fake
        _prepare(tmp_path, exporter)
        assert prepare_uma_aoti(HCN, cache_dir=tmp_path, env=ENV, exporter=exporter).hit

    def test_miss_without_structure_is_an_error(self, tmp_path: Path, fake):
        exporter, _count = fake
        with pytest.raises(ValueError, match="cache miss"):
            prepare_uma_aoti(HCN, cache_dir=tmp_path, env=ENV, exporter=exporter)

    def test_other_composition_misses(self, tmp_path: Path, fake, monkeypatch):
        exporter, count = fake
        _prepare(tmp_path, exporter)
        monkeypatch.setenv("FAKE_Z", "6,6,7,7,1,1")
        doubled = prepare_uma_aoti(
            [6, 6, 7, 7, 1, 1],
            cache_dir=tmp_path,
            exporter=exporter,
            env=ENV,
            atoms_path=exporter,
        )
        assert not doubled.hit
        assert _calls(count) == 2
        assert len(entries(tmp_path)) == 2

    def test_dry_run_compiles_nothing(self, tmp_path: Path, fake):
        exporter, count = fake
        res = _prepare(tmp_path, exporter, dry_run=True, molecular_box=25.0, batch_max=8)
        assert not res.hit
        assert res.compile_seconds is None
        assert res.command[-4:] == ["--molecular-box", "25", "--batch-max", "8"]
        assert _calls(count) == 0
        assert not res.path.exists()

    def test_force_recompiles(self, tmp_path: Path, fake):
        exporter, count = fake
        _prepare(tmp_path, exporter)
        res = _prepare(tmp_path, exporter, force=True)
        assert not res.hit
        assert _calls(count) == 2
        assert lookup(tmp_path, res.key) == res.path

    def test_exporter_failure_installs_nothing(self, tmp_path: Path, fake, monkeypatch):
        exporter, _count = fake
        monkeypatch.setenv("FAKE_EXIT", "4")
        with pytest.raises(RuntimeError, match="exited 4: the compiled AOTI package"):
            _prepare(tmp_path, exporter)
        assert entries(tmp_path) == []
        assert not list(tmp_path.glob(".partial-*"))

    def test_package_that_disagrees_with_its_key_is_refused(
        self, tmp_path: Path, fake, monkeypatch
    ):
        exporter, _count = fake
        monkeypatch.setenv("FAKE_SPIN", "3")
        with pytest.raises(RuntimeError, match="does not match its key: spin"):
            _prepare(tmp_path, exporter)
        assert entries(tmp_path) == []

    def test_stale_partial_is_dropped(self, tmp_path: Path, fake):
        exporter, _count = fake
        key = aoti_key(HCN, ENV)
        stale = tmp_path / f".partial-{key.stem()}-dead"
        stale.mkdir()
        (stale / "half.pt2").write_bytes(b"x")
        assert stale_partials(tmp_path) == [stale]
        _prepare(tmp_path, exporter)
        assert not stale.exists()

    def test_concurrent_preparations_compile_once(
        self, tmp_path: Path, fake, monkeypatch
    ):
        exporter, count = fake
        monkeypatch.setenv("FAKE_SLEEP", "1.0")
        cache = tmp_path / "cache"
        procs = [_child("prepare", cache, exporter) for _ in range(4)]
        results = [_finish(p) for p in procs]
        assert _calls(count) == 1
        assert len({path for path, _hit in results}) == 1
        assert sorted(hit for _path, hit in results) == ["False", "True", "True", "True"]

    def test_locked_entry_is_not_removed(self, tmp_path: Path, fake):
        exporter, _count = fake
        res = _prepare(tmp_path, exporter)
        entry = find_entry(tmp_path, res.key.digest())
        with key_lock(tmp_path, res.key.stem()):
            assert _finish(_child("remove", tmp_path, entry.stem)) == ["False"]
        assert remove(tmp_path, entry)
        assert entries(tmp_path) == []


# Separate processes, as concurrent prepare-aoti runs are.
CHILD = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from rgpycrumbs.uma import ExportEnv, find_entry, prepare_uma_aoti, remove

    action, cache, arg = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if action == "prepare":
        env = ExportEnv(*sys.argv[4].split(";"))
        res = prepare_uma_aoti(
            [6, 7, 1], cache_dir=cache, exporter=Path(arg), env=env, atoms_path=Path(arg)
        )
        print(res.path, res.hit)
    else:
        print(remove(cache, find_entry(cache, arg)))
    """
)


def _child(action: str, cache: Path, arg) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), *sys.path])}
    packed = ";".join([ENV.device, ENV.target, ENV.torch, ENV.fairchem])
    return subprocess.Popen(
        [sys.executable, "-c", CHILD, action, str(cache), str(arg), packed],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _finish(proc: subprocess.Popen) -> list[str]:
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, err
    return out.split()


class TestFindEntry:
    def test_prefix_digest_and_ambiguity(self, tmp_path: Path, fake):
        exporter, _count = fake
        a = _prepare(tmp_path, exporter)
        b = _prepare(tmp_path, exporter, charge=1, spin=2)
        assert find_entry(tmp_path, a.key.digest()).path == a.path
        assert find_entry(tmp_path, b.key.stem()).path == b.path
        assert find_entry(tmp_path, "omol-CHN-q1").path == b.path
        with pytest.raises(LookupError, match="matches 2 entries"):
            find_entry(tmp_path, "omol-CHN")
        with pytest.raises(LookupError, match="no cache entry"):
            find_entry(tmp_path, "omat")


PROBE_PYTHON = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    with open(os.environ["PROBE_COUNT"], "a") as fh:
        fh.write(" ".join(sys.argv[1:]) + "\\n")
    if os.environ.get("PROBE_ERROR"):
        print(json.dumps({{"error": os.environ["PROBE_ERROR"]}}))
        sys.exit(0)
    site = os.environ["PROBE_SITE"]
    print(json.dumps({{
        "device": sys.argv[2].split(":")[0],
        "target": "x86_64|AVX2|",
        "torch": "2.8.0",
        "fairchem": "2.21.0",
        "stamps": {{site: os.stat(site).st_mtime_ns}},
    }}))
    """
)


class TestProbeEnv:
    @pytest.fixture
    def probe(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        python = tmp_path / "python"
        python.write_text(PROBE_PYTHON.format(python=sys.executable))
        python.chmod(0o755)
        site = tmp_path / "site-packages"
        site.mkdir()
        count = tmp_path / "probes.txt"
        count.write_text("")
        monkeypatch.setenv("PROBE_COUNT", str(count))
        monkeypatch.setenv("PROBE_SITE", str(site))
        monkeypatch.delenv("PROBE_ERROR", raising=False)
        return python, site, count

    def test_probe_is_remembered_until_site_packages_change(self, tmp_path, probe):
        python, site, count = probe
        cache = tmp_path / "cache"
        env = probe_env(cache, python=str(python))
        assert env == ExportEnv("cpu", "x86_64|AVX2|", "2.8.0", "2.21.0")
        assert probe_env(cache, python=str(python)) == env
        assert _calls(count) == 1
        (site / "new_dist-1.0.dist-info").mkdir()
        os.utime(site, ns=(1, 1))
        probe_env(cache, python=str(python))
        assert _calls(count) == 2

    def test_running_interpreter_is_keyed_by_what_is_installed(
        self, tmp_path, probe, monkeypatch
    ):
        # uv run --script can put one package set at a new path on every
        # run; two such environments share the memo.
        python, _site, count = probe
        site = tmp_path / "env-site"
        site.mkdir()
        (site / "torch-2.13.0.dist-info").mkdir()
        monkeypatch.setattr(sys, "path", [str(site)])
        first = tmp_path / "env-a" / "python"
        second = tmp_path / "env-b" / "python"
        for link in (first, second):
            link.parent.mkdir()
            link.symlink_to(python)
        cache = tmp_path / "cache"
        monkeypatch.setattr(sys, "executable", str(first))
        probe_env(cache)
        monkeypatch.setattr(sys, "executable", str(second))
        probe_env(cache)
        assert _calls(count) == 1
        (site / "fairchem_core-2.23.0.dist-info").mkdir()
        probe_env(cache)
        assert _calls(count) == 2

    def test_devices_are_probed_separately(self, tmp_path, probe):
        python, _site, count = probe
        assert probe_env(tmp_path, python=str(python), device="cuda").device == "cuda"
        probe_env(tmp_path, python=str(python), device="cpu")
        assert _calls(count) == 2

    def test_probe_error_is_reported(self, tmp_path, probe, monkeypatch):
        python, _site, _count = probe
        monkeypatch.setenv("PROBE_ERROR", "fairchem-core is not installed")
        with pytest.raises(RuntimeError, match="fairchem-core is not installed"):
            probe_env(tmp_path, python=str(python))

    def test_target_fields_agree(self):
        # _probe imports torch, so its tuple is read from the source.
        import ast

        from rgpycrumbs.uma import _key

        tree = ast.parse(Path(_key.__file__).with_name("_probe.py").read_text())
        probe_fields = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign) and node.targets[0].id == "TARGET_FIELDS"
        )
        assert probe_fields == _key.TARGET_FIELDS

    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "CUDA"])
    def test_devices(self, device):
        assert normalize_device(device) == device.lower()

    @pytest.mark.parametrize("device", ["gpu", "cuda:x", "cpu:0", "mps"])
    def test_bad_devices(self, device):
        with pytest.raises(ValueError, match="device must be"):
            normalize_device(device)


class TestResolveExporter:
    def test_env_and_explicit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        script = tmp_path / "export_uma_aoti.py"
        script.write_text("# exporter\n")
        monkeypatch.delenv("RGPOT_EXPORT_UMA", raising=False)
        assert resolve_exporter(script) == script.resolve()
        monkeypatch.setenv("RGPOT_EXPORT_UMA", str(script))
        assert resolve_exporter() == script.resolve()

    def test_env_naming_no_file_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RGPOT_EXPORT_UMA", str(tmp_path / "missing.py"))
        with pytest.raises(FileNotFoundError, match="RGPOT_EXPORT_UMA"):
            resolve_exporter()

    def test_rgpot_checkout_is_found_from_a_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RGPOT_EXPORT_UMA", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path / "nobin"))
        script = tmp_path / "scripts" / "export_uma_aoti.py"
        script.parent.mkdir()
        script.write_text("# exporter\n")
        work = tmp_path / "runs" / "neb"
        work.mkdir(parents=True)
        monkeypatch.chdir(work)
        assert resolve_exporter() == script
