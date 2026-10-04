# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rgoswami@ieee.org>
# SPDX-License-Identifier: MIT
"""On-disk cache of UMA AOTI packages.

Layout under the cache directory::

    <stem>.pt2         AOTInductor package, the file rgpot's UmaPot loads
    <stem>.pt2.json    sidecar: the full key, the package's embedded
                       metadata, compile time and provenance
    .locks/<stem>.lock per-key lock held while the package is prepared
    .partial-<stem>-*/ exporter output before it is installed

``<stem>`` comes from :meth:`AotiKey.stem`. A lookup is one ``stat`` and
one sidecar read: no scan of the directory. An entry is installed by
renaming the finished package, then its sidecar, into place, so a
reader never sees a partial ``.pt2`` and a sidecar always names a
complete one.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import fnmatch
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rgpycrumbs.uma._key import TARGET_FIELDS, AotiKey, canonical_float

SIDECAR_SUFFIX = ".pt2.json"
_LOCK_DIR = ".locks"
_PARTIAL_PREFIX = ".partial-"
_EMBEDDED_PATTERN = "*/data/aotinductor/*/*_metadata.json"


def default_cache_dir() -> Path:
    """``$RGPYCRUMBS_UMA_CACHE``, else ``$XDG_CACHE_HOME/rgpycrumbs/uma``."""
    override = os.environ.get("RGPYCRUMBS_UMA_CACHE", "").strip()
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "rgpycrumbs" / "uma"


def package_path(cache_dir: Path, key: AotiKey) -> Path:
    return Path(cache_dir) / f"{key.stem()}.pt2"


def sidecar_path(pt2: Path) -> Path:
    pt2 = Path(pt2)
    return pt2.with_name(pt2.name + ".json")


def read_sidecar(pt2: Path) -> dict[str, Any] | None:
    """Return the sidecar of ``pt2``, or None when it is absent or unreadable."""
    try:
        data = json.loads(sidecar_path(pt2).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def lookup(cache_dir: Path, key: AotiKey) -> Path | None:
    """Return the cached package for ``key``, or None on a miss.

    A hit needs the package and a sidecar whose stored key equals
    ``key`` field by field, so a digest collision can never return the
    wrong package.
    """
    pt2 = package_path(cache_dir, key)
    if not pt2.is_file():
        return None
    data = read_sidecar(pt2)
    if data is None or data.get("key") != key.as_dict():
        return None
    return pt2


# --------------------------------------------------------------- metadata


def embedded_metadata(pt2: Path) -> dict[str, Any]:
    """Metadata torch embedded in the package, as rgpot's loader reads it.

    The exporter passes its runtime contract through
    ``aot_inductor.metadata``; torch stores it, plus its own
    ``AOTI_*`` device fields, as string values in
    ``<name>/data/aotinductor/<model>/*_metadata.json``. This reads the
    archive directly, so it needs no torch.
    """
    with zipfile.ZipFile(pt2) as archive:
        names = sorted(
            n for n in archive.namelist() if fnmatch.fnmatch(n, _EMBEDDED_PATTERN)
        )
        for name in names:
            data = json.loads(archive.read(name))
            if isinstance(data, dict) and data:
                return data
    raise ValueError(f"{pt2}: no AOTI metadata json in the package")


def _literal(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def embedded_mismatches(meta: Mapping[str, Any], key: AotiKey) -> list[str]:
    """Fields where a package's embedded metadata disagrees with ``key``.

    Embedded values are strings (torch stores ``str(value)``). Every
    package records ``z_set`` and the traced input shapes. rgpot's
    exporter also embeds ``natoms``, the per-element ``counts`` (JSON of
    atomic number to count), and the ``model``, ``torch_version`` and
    ``fairchem_version`` that compiled it; rgpot's ``UmaPot`` refuses an
    input whose counts differ. Those fields are checked when present, so
    a package from an older exporter still passes on the element set and
    the traced atom count.
    """
    out: list[str] = []

    def check(field: str, got: Any, want: Any) -> None:
        if got != want:
            out.append(f"{field}: package {got!r}, key {want!r}")

    def as_int(field: str) -> Any:
        try:
            return int(str(meta.get(field)))
        except ValueError:
            return meta.get(field)

    check("task_name", meta.get("task_name"), key.task)
    check("charge", as_int("charge"), key.charge)
    check("spin", as_int("spin"), key.spin)
    check("z_set", _literal(meta.get("z_set")), key.z_set)
    shapes = _literal(meta.get("shapes"))
    systems = key.batch_max if key.batch_max > 1 else 1
    natoms = None
    if isinstance(shapes, dict) and shapes.get("pos"):
        natoms = int(shapes["pos"][0]) // systems
    check("natoms", natoms, key.natoms)
    if "natoms" in meta:
        check("embedded natoms", as_int("natoms"), key.natoms)
    if "counts" in meta:
        counts = _literal(meta.get("counts"))
        if isinstance(counts, str):
            try:
                counts = json.loads(counts)
            except ValueError:
                pass
        if isinstance(counts, dict):
            try:
                counts = tuple(sorted((int(z), int(n)) for z, n in counts.items()))
            except (TypeError, ValueError):
                pass
        check("counts", counts, key.counts)
    for field, want in (
        ("model", key.model),
        ("torch_version", key.torch),
        ("fairchem_version", key.fairchem),
    ):
        if field in meta:
            check(field, meta.get(field), want)
    check("pos_dtype", meta.get("pos_dtype"), key.dtype)
    box = meta.get("molecular_box")
    check(
        "molecular_box",
        canonical_float(box) if box is not None else None,
        key.molecular_box,
    )
    check("batch_max", as_int("batch_max") if "batch_max" in meta else 0, key.batch_max)
    check("AOTI_DEVICE_KEY", meta.get("AOTI_DEVICE_KEY"), key.device)
    target = "|".join(str(meta.get(f, "")) for f in TARGET_FIELDS)
    check("target", target, key.target)
    return out


# ------------------------------------------------------------------ locks


@contextlib.contextmanager
def key_lock(cache_dir: Path, stem: str, *, wait: bool = True) -> Iterator[bool]:
    """Hold the exclusive lock of one cache entry.

    Yields True once the lock is held. With ``wait=False`` it yields
    False instead of blocking when another process holds it. The lock is
    ``flock`` on ``.locks/<stem>.lock``; the kernel drops it when the
    holder exits, so a killed preparation never leaves the key locked.
    """
    lock_dir = Path(cache_dir) / _LOCK_DIR
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_dir / f"{stem}.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                yield False
                return
            print(f"waiting for another preparation of {stem}", file=sys.stderr)
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def partial_dir(cache_dir: Path, stem: str) -> Path:
    """Fresh private directory for one exporter run, after dropping stale ones.

    Call with the key lock held: any older ``.partial-<stem>-*`` then
    belongs to a run that died.
    """
    cache_dir = Path(cache_dir)
    for stale in cache_dir.glob(f"{_PARTIAL_PREFIX}{stem}-*"):
        shutil.rmtree(stale, ignore_errors=True)
    return Path(tempfile.mkdtemp(prefix=f"{_PARTIAL_PREFIX}{stem}-", dir=cache_dir))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    umask = os.umask(0)
    os.umask(umask)
    # mkstemp creates 0600; the sidecar is as readable as its package.
    os.fchmod(fd, 0o666 & ~umask)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def install(
    cache_dir: Path, key: AotiKey, built: Path, sidecar: Mapping[str, Any]
) -> Path:
    """Move a finished package into the cache with its sidecar.

    ``built`` must be on the cache's filesystem (inside
    :func:`partial_dir`), so the rename is atomic. The package lands
    first and the sidecar second; :func:`lookup` treats a package
    without a matching sidecar as a miss.
    """
    dest = package_path(cache_dir, key)
    with contextlib.suppress(FileNotFoundError):
        sidecar_path(dest).unlink()
    with open(built, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(built, dest)
    _write_json_atomic(sidecar_path(dest), {**sidecar, "key": key.as_dict()})
    return dest


# ---------------------------------------------------------------- entries


@dataclass(frozen=True)
class CacheEntry:
    """One package in the cache and what its sidecar says about it."""

    path: Path
    sidecar: dict[str, Any] | None

    @property
    def key(self) -> AotiKey | None:
        raw = (self.sidecar or {}).get("key")
        if not isinstance(raw, dict):
            return None
        try:
            return AotiKey.from_dict(raw)
        except (TypeError, ValueError):
            return None

    @property
    def stem(self) -> str:
        return self.path.name[: -len(".pt2")]

    @property
    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


def entries(cache_dir: Path) -> list[CacheEntry]:
    """Every ``.pt2`` in the cache, sorted by name."""
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    return [CacheEntry(p, read_sidecar(p)) for p in sorted(root.glob("*.pt2"))]


def find_entry(cache_dir: Path, name: str) -> CacheEntry:
    """Resolve a path, a file name, a stem, or a unique stem prefix or digest."""
    path = Path(name)
    if path.suffix == ".pt2" and path.is_file():
        return CacheEntry(path, read_sidecar(path))
    found = [
        e
        for e in entries(cache_dir)
        if name in (e.path.name, e.stem)
        or e.stem.startswith(name)
        or e.stem.endswith(f"-{name}")
    ]
    exact = [e for e in found if name in (e.stem, e.path.name)]
    if exact:
        return exact[0]
    if len(found) == 1:
        return found[0]
    if not found:
        raise LookupError(f"no cache entry matches {name!r} in {cache_dir}")
    names = ", ".join(e.stem for e in found)
    raise LookupError(f"{name!r} matches {len(found)} entries: {names}")


def remove(cache_dir: Path, entry: CacheEntry, *, wait: bool = False) -> bool:
    """Delete one entry under its key lock. False when it is being prepared."""
    with key_lock(cache_dir, entry.stem, wait=wait) as held:
        if not held:
            return False
        with contextlib.suppress(FileNotFoundError):
            sidecar_path(entry.path).unlink()
        with contextlib.suppress(FileNotFoundError):
            entry.path.unlink()
    # The lock file stays: unlinking it would let a waiter on the old
    # inode and a newcomer on a fresh one both hold "the" lock.
    return True


def stale_partials(cache_dir: Path) -> list[Path]:
    """Partial exporter directories whose key is not locked by a live run."""
    root = Path(cache_dir)
    out = []
    for path in sorted(root.glob(f"{_PARTIAL_PREFIX}*")):
        stem = path.name[len(_PARTIAL_PREFIX) :].rsplit("-", 1)[0]
        with key_lock(root, stem, wait=False) as held:
            if held:
                out.append(path)
    return out
