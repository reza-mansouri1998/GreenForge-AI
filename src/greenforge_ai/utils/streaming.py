"""Bounded SQL, atomic files, locks, hashes and Parquet validation."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import uuid
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@contextlib.contextmanager
def connection(temp: Path, memory="1GB", threads=2):
    temp.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.execute("SET memory_limit=" + literal(memory))
        con.execute("SET temp_directory=" + literal(temp))
        con.execute(f"SET threads={int(threads)}")
        con.execute("SET preserve_insertion_order=false")
        yield con


@contextlib.contextmanager
def atomic_path(path: Path):
    """Publish a closed temporary file only after syncing it successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield temp
        # Windows fsync uses _commit, which requires a writable descriptor.
        # r+b preserves the completed file without truncating its contents.
        with temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temp:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def pipeline_lock(path):
    """OS locks release automatically on process exit, including a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another ETL run holds the project lock") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_parquet(path, dense=False, step_ms=None):
    previous, rows, nulls = None, 0, {}
    with pq.ParquetFile(path) as source:
        for batch in source.iter_batches(batch_size=16384):
            stamps = batch.column(batch.schema.get_field_index("WsDateTime"))
            if stamps.null_count:
                raise ValueError("Null timestamp")
            # Both timezone-naive publisher wall-clock and UTC timestamps are
            # represented as integer milliseconds. Never localize the former.
            ticks = pc.cast(stamps, pa.int64()).to_numpy()
            differences = np.diff(ticks if previous is None else np.r_[previous, ticks])
            if np.any(differences <= 0) or (step_ms is not None and np.any(differences != step_ms)):
                raise ValueError("Duplicated, unsorted, or off-grid timestamps")
            if len(ticks):
                previous = ticks[-1]
            for name, arr in zip(batch.schema.names, batch.columns):
                nulls[name] = nulls.get(name, 0) + arr.null_count
                if dense and arr.null_count:
                    raise ValueError(f"Missing ML values: {name}")
                if (
                    pa.types.is_floating(arr.type)
                    and not pc.all(pc.fill_null(pc.is_finite(arr), True)).as_py()
                ):
                    raise ValueError(f"NaN/infinity in {name}; missing values must be null")
            rows += batch.num_rows
    return {"rows": rows, "null_counts": nulls}


def recover_publication(processed: Path):
    """Roll back an interrupted stable-path publication before another ETL run."""
    journal = processed / ".publication.json"
    if not journal.exists():
        return
    state = json.loads(journal.read_text(encoding="utf-8"))
    backup = processed / state["backup"]
    for entry in reversed(state["entries"]):
        target = processed / entry["path"]
        saved = backup / entry["path"]
        if saved.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(saved, target)
        elif not entry["existed"]:
            target.unlink(missing_ok=True)
    journal.unlink()
    import shutil

    shutil.rmtree(backup, ignore_errors=True)


def publish_files(processed: Path, staged: dict[Path, Path | None]):
    """Replace stable files with rollback on failure and a crash recovery journal.

    A None source removes a stale artifact. Call while holding the ETL lock.
    The manifest must be the last mapping entry. Readers must not run during
    publication because multiple filesystem paths cannot be replaced atomically.
    """
    import shutil

    backup = processed / f".backup_{uuid.uuid4().hex}"
    backup.mkdir()
    journal = processed / ".publication.json"
    entries = [{"path": str(target.relative_to(processed)), "existed": target.exists()}
               for target in staged]
    write_json(journal, {"backup": backup.name, "entries": entries})
    try:
        for target, source in staged.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                saved = backup / target.relative_to(processed)
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, saved)
            if source is not None:
                os.replace(source, target)
    except BaseException:
        recover_publication(processed)
        raise
    else:
        journal.unlink()  # Commit: a later run must not roll back these outputs.
        shutil.rmtree(backup, ignore_errors=True)
