"""Create a portable MLflow SQLite snapshot for Linux from the host Windows DB.

The source tracking DB is opened read-only. Only the snapshot's file: artifact
URIs are rewritten; no model files or historical reports are copied/modified.
This is for local retrospective replay, not concurrent training in containers.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote, unquote
import uuid

SOURCE = Path(os.getenv("GREENFORGE_SOURCE_MLFLOW_DB", "/source/tracking.db"))
DEST = Path(os.getenv("GREENFORGE_PORTABLE_MLFLOW_DB", "/app/mlruns/tracking.db"))
ARTIFACT_DIR = "/app/mlflow_artifacts"

# Match only a known, local MLflow artifact path, never remote storage URIs.
ARTIFACT_MARKER = re.compile(r"/mlruns/artifacts(?=/|$)", re.IGNORECASE)
COLUMNS = (
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "source"),
    ("model_versions", "storage_location"),
)


def rebase_file_uri(uri: str | None) -> str | None:
    """Map a Windows artifact path to its bind-mounted Linux location."""
    if not isinstance(uri, str) or not uri.startswith("file:"):
        return uri
    decoded = unquote(uri).replace("\\", "/")
    marker = ARTIFACT_MARKER.search(decoded)
    if marker is None:
        return uri  # For example, an unrelated MLflow Default experiment.
    suffix = decoded[marker.end():].lstrip("/")
    segments = [part for part in suffix.split("/") if part]
    if any(part in {".", ".."} for part in segments):
        raise ValueError("Unsafe path segment in MLflow artifact URI")
    return "file://" + ARTIFACT_DIR + ("/" + quote("/".join(segments), safe="/") if segments else "")


def make_portable_db(source: Path = SOURCE, dest: Path = DEST) -> int:
    if not source.is_file():
        raise FileNotFoundError(f"Missing host MLflow database: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    changed = 0
    try:
        # SQLite backup yields a transactionally consistent copy even if the
        # source DB is in use. The host file is never opened for writing.
        src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True, timeout=20)
        try:
            dst = sqlite3.connect(temp)
            try:
                src.backup(dst)
                present = {
                    row[0] for row in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                with dst:
                    for table, column in COLUMNS:
                        if table not in present:
                            continue
                        columns = {row[1] for row in dst.execute(f"PRAGMA table_info({table})")}
                        if column not in columns:
                            continue
                        rows = dst.execute(
                            f"SELECT rowid, {column} FROM {table} WHERE {column} LIKE 'file:%'"
                        ).fetchall()
                        for rowid, original in rows:
                            updated = rebase_file_uri(original)
                            if updated != original:
                                dst.execute(
                                    f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                                    (updated, rowid),
                                )
                                changed += 1
                # Ensure the source was a functioning SQL-backed tracking DB.
                if not {"experiments", "runs", "logged_models"}.issubset(present):
                    raise ValueError("Input is not a compatible MLflow SQLite tracking database")
            finally:
                dst.close()
        finally:
            src.close()
        os.chmod(temp, 0o664)
        os.replace(temp, dest)
        # API runs as UID 10001; only the COPY is writable to containers.
        if os.geteuid() == 0:
            os.chown(dest.parent, 10001, 10001)
            os.chown(dest, 10001, 10001)
    finally:
        temp.unlink(missing_ok=True)
    return changed


if __name__ == "__main__":
    count = make_portable_db()
    print(f"Portable MLflow DB prepared: {DEST} ({count} artifact locations rewritten)")
