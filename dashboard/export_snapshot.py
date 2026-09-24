#!/usr/bin/env python3
"""Export an allowlisted beacon snapshot and capture catalog for static hosting.

This command only writes local files. Upload the resulting JSON gzip snapshots
and catalog to an object host such as Cloudflare R2 separately; source DuckDB
databases and their provenance tables are never included in an export.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import duckdb


MAX_ROWS = 500_000
# Keep these in sync with the browser snapshot loader in data.mjs.
MAX_COMPRESSED_BYTES = 100 * 1024 * 1024
MAX_EXPANDED_BYTES = 400 * 1024 * 1024
SNAPSHOT_FORMAT = "suncet-beacon-v1"
CATALOG_FORMAT = "suncet-catalog-v1"
ALLOWLIST_PATH = Path(__file__).with_name("public-fields.json")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("Invalid database identifier")
    return '"' + value.replace('"', '""') + '"'


def _qualified(schema: str, table: str) -> str:
    return f"{_identifier(schema)}.{_identifier(table)}"


def _dataset_id(title: str, supplied: str | None) -> str:
    if supplied is not None:
        if not SAFE_ID.fullmatch(supplied):
            raise ValueError("--id must be 1–128 ASCII letters, digits, hyphens, or underscores, starting with a letter or digit")
        return supplied
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:128].rstrip("-")
    if not slug:
        raise ValueError("The title cannot form a safe filename; supply --id explicitly")
    return slug


def _allowlist() -> list[str]:
    fields = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    if not isinstance(fields, list) or not fields or any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("public-fields.json must be a nonempty array of field names")
    if len(fields) != len(set(fields)):
        raise ValueError("public-fields.json contains duplicate fields")
    return fields


def _beacon_table(connection) -> tuple[str, list[str]]:
    tables = connection.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_type = 'BASE TABLE' "
        "AND table_schema NOT IN ('information_schema', 'pg_catalog') "
        "ORDER BY table_schema, table_name"
    ).fetchall()
    schema_rows = connection.execute(
        "SELECT table_schema, table_name, column_name FROM information_schema.columns "
        "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') "
        "ORDER BY table_schema, table_name, ordinal_position"
    ).fetchall()
    columns = {}
    for schema, table, field in schema_rows:
        columns.setdefault((schema, table), []).append(field)

    catalog_entries = []
    for schema, table in tables:
        if table == "_apid_catalog":
            catalog_entries.extend(connection.execute(
                f"SELECT apid, packet_name, table_name FROM {_qualified(schema, table)}"
            ).fetchall())
    candidates = []
    for schema, table in tables:
        names = columns.get((schema, table), [])
        score = sum(field.startswith("beac_") for field in names)
        for apid, packet_name, table_name in catalog_entries:
            if table_name == table and (apid == 1 or "beacon" in str(packet_name).lower()):
                score += 20_000 if apid == 1 else 10_000
        if score:
            candidates.append((score, schema, table, names))
    if not candidates:
        raise ValueError("No beacon telemetry table found; the database needs APID 1 or beac_* columns")
    _, schema, table, names = max(candidates, key=lambda candidate: candidate[0])
    return _qualified(schema, table), names


def _public_value(value):
    """Keep JSON scalars, preserving missing readings without creating NaN JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, Decimal)):
        result = float(value)
        return result if math.isfinite(result) else None
    # Beacon channels are scalar. Unexpected complex/UUID values are not exposed.
    return None


def _created_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _read_catalog(path: Path) -> dict:
    if not path.exists():
        return {"format": CATALOG_FORMAT, "datasets": []}
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or catalog.get("format") != CATALOG_FORMAT or not isinstance(catalog.get("datasets"), list):
        raise ValueError(f"Existing catalog is not a {CATALOG_FORMAT} catalog: {path}")
    if any(not isinstance(entry, dict) or not isinstance(entry.get("id"), str) for entry in catalog["datasets"]):
        raise ValueError("Existing catalog has an invalid dataset entry")
    ids = [entry["id"] for entry in catalog["datasets"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Existing catalog has duplicate dataset IDs")
    return catalog


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_snapshot(
    database_path: str | Path,
    output_dir: str | Path,
    title: str,
    *,
    dataset_id: str | None = None,
    reset: int | None = None,
) -> dict:
    """Write one compressed capture and atomically upsert its catalog entry."""
    source = Path(database_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Database file does not exist: {source}")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("A nonempty capture title is required")
    title = title.strip()
    slug = _dataset_id(title, dataset_id)
    output = Path(output_dir).expanduser().resolve()
    catalog_path = output / "catalog.json"
    catalog = _read_catalog(catalog_path)

    with duckdb.connect(str(source), read_only=True) as connection:
        table, available = _beacon_table(connection)
        selected = [field for field in _allowlist() if field in available]
        if not selected or not any(field.startswith("beac_") for field in selected):
            raise ValueError("The beacon table has no recognized telemetry channels")
        parameters = []
        where = ""
        if reset is not None:
            if "beac_num_sc_resets" not in available:
                raise ValueError("This database has no beac_num_sc_resets field to filter")
            where = ' WHERE "beac_num_sc_resets" = ?'
            parameters = [reset]
        row_count = connection.execute(f"SELECT count(*) FROM {table}{where}", parameters).fetchone()[0]
        if not row_count:
            raise ValueError("No beacon rows match this selection")
        if row_count > MAX_ROWS:
            raise ValueError(f"Selection contains {row_count:,} rows; the limit is {MAX_ROWS:,}. Use --reset to select a smaller capture")
        order = ' ORDER BY "packet_index"' if "packet_index" in available else ""
        records = connection.execute(
            f"SELECT {', '.join(_identifier(field) for field in selected)} FROM {table}{where}{order} LIMIT {MAX_ROWS + 1}",
            parameters,
        ).fetchall()
        if len(records) > MAX_ROWS:
            raise ValueError(f"Selection exceeds the {MAX_ROWS:,}-row limit")

    snapshot = {
        "format": SNAPSHOT_FORMAT,
        "title": title,
        "createdUtc": _created_utc(),
        "columns": selected,
        "rows": [[_public_value(value) for value in record] for record in records],
        "rowCount": len(records),
    }
    expanded = _json_bytes(snapshot)
    if len(expanded) > MAX_EXPANDED_BYTES:
        raise ValueError(
            f"Snapshot JSON is {len(expanded):,} bytes; the expanded-size limit is "
            f"{MAX_EXPANDED_BYTES:,} bytes. Use --reset to select a smaller capture"
        )
    compressed = gzip.compress(expanded, compresslevel=9, mtime=0)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ValueError(
            f"Compressed snapshot is {len(compressed):,} bytes; the download-size limit is "
            f"{MAX_COMPRESSED_BYTES:,} bytes. Use --reset to select a smaller capture"
        )
    filename = f"{slug}.json.gz"
    entry = {"id": slug, "title": title, "url": filename, "rowCount": len(records), "bytes": len(compressed)}
    for index, existing in enumerate(catalog["datasets"]):
        if existing["id"] == slug:
            catalog["datasets"][index] = entry
            break
    else:
        catalog["datasets"].append(entry)

    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(output / filename, compressed)
    _atomic_write(catalog_path, _json_bytes(catalog) + b"\n")
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="A completed SunCET DuckDB snapshot (opened read-only)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Explicit local export directory; no files are uploaded")
    parser.add_argument("--title", required=True, help="Public capture title shown in the dashboard")
    parser.add_argument("--id", dest="dataset_id", help="Stable safe capture ID; defaults to a slug of the title")
    parser.add_argument("--reset", type=int, help="Export only this spacecraft reset-counter value")
    arguments = parser.parse_args(argv)
    try:
        result = export_snapshot(arguments.database, arguments.output_dir, arguments.title,
                                 dataset_id=arguments.dataset_id, reset=arguments.reset)
    except (ValueError, OSError, duckdb.Error) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(json.dumps({"outputDir": str(arguments.output_dir.resolve()), "dataset": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
