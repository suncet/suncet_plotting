"""Select all ingested batches belonging to one capture, in ingestion order."""
from pathlib import Path
from export_snapshot import _identifier


def capture_selection(connection, folder, apid, schema=None):
    prefix = f'{_identifier(schema)}.' if schema else ''
    ingestion_table = prefix + '_ingestions'
    fields = {r[0] for r in connection.execute(f'DESCRIBE {ingestion_table}').fetchall()}
    order = 'ingested_utc, source_path' if 'ingested_utc' in fields else 'source_path'
    all_rows = connection.execute(f'SELECT source_path,source_sha256,rows_ingested,table_name FROM {ingestion_table} WHERE apid=? ORDER BY {order}', [apid]).fetchall()
    matches = [r for r in all_rows if Path(r[0]).resolve().parent == Path(folder).resolve() / 'decoded_packets']
    if not matches:
        return None, []
    if len({r[3] for r in matches}) != 1:
        raise ValueError(f'APID {apid} batches reference different tables.')
    table = prefix + _identifier(matches[0][3])
    literal = lambda s: "'" + s.replace("'", "''") + "'"
    hashes = ','.join(literal(r[1]) for r in matches)
    counts = connection.execute(f'SELECT _source_sha256,count(*),count(DISTINCT packet_index) FROM {table} WHERE _source_sha256 IN ({hashes}) GROUP BY 1').fetchall()
    actual = {h: (count, unique) for h, count, unique in counts}
    for path, digest, count, _ in matches:
        if actual.get(digest, (0, 0)) != (count, count):
            raise ValueError(f'APID {apid} batch has mismatched counts or duplicate packet indices: {Path(path).name}')
    # Packet indices restart for each input. Preserve them and give the combined
    # capture a unique display index; no onboard timestamps are changed here.
    rank = 'CASE _source_sha256 ' + ' '.join(f'WHEN {literal(r[1])} THEN {i}' for i,r in enumerate(matches)) + ' END'
    query = (f'SELECT row_number() OVER (ORDER BY {rank},packet_index)-1 AS packet_index, '
             f'packet_index AS original_packet_index, * EXCLUDE(packet_index) '
             f'FROM {table} WHERE _source_sha256 IN ({hashes})')
    return query, matches
