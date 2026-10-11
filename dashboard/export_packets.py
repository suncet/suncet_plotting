"""Export explicitly allowlisted engineering packets, scoped to one capture.

Keep original spacecraft times. Unset times remain visible on packet order,
and are omitted on UTC. No beacon time correction is borrowed across packets.
"""
import gzip
import json
from pathlib import Path

import duckdb
from export_snapshot import (_identifier, _json_bytes, _public_value, _created_utc,
                             _atomic_write, MAX_ROWS, MAX_EXPANDED_BYTES, MAX_COMPRESSED_BYTES)

DEFINITIONS = Path(__file__).with_name('public-packets.json')
CONTEXT = ['packet_index', 'sequence_count', 'decode_status', 'is_duplicate_packet']


def export_packets(database, capture_folder, output_dir, dataset_id):
    definitions = json.loads(DEFINITIONS.read_text())
    output = Path(output_dir)
    capture = Path(capture_folder).resolve()
    packets = []
    with duckdb.connect(str(database), read_only=True) as c:
        ingestions = c.execute('SELECT apid,table_name,source_path,source_sha256,rows_ingested FROM _ingestions').fetchall()
        for definition in definitions:
            if definition['name'].lower().startswith('adcs_'):
                continue
            matches = [r for r in ingestions if r[0] == definition['apid'] and Path(r[2]).resolve().parent == capture / 'decoded_packets']
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError(f"Multiple ingestions for APID {definition['apid']}; inspect before publishing.")
            apid, table, _, source_hash, row_count = matches[0]
            if not row_count:
                continue
            if row_count > MAX_ROWS:
                raise ValueError(f'APID {apid} exceeds {MAX_ROWS:,} rows. Split the capture before publishing.')
            fields = {r[0] for r in c.execute(f'DESCRIBE {_identifier(table)}').fetchall()}
            signals = [s for s in definition['signals'] if s['field'] in fields]
            if not signals:
                continue
            sec, sub = f"ccsdsSecHeader2_sec_{definition['name']}", f"ccsdsSecHeader2_sub_{definition['name']}"
            selected = [f for f in CONTEXT if f in fields] + [s['field'] for s in signals]
            expressions = [_identifier(f) for f in selected]
            if sec in fields and sub in fields:
                selected += ['packet_time_sec', 'packet_time_sub']
                expressions += [f'{_identifier(sec)} AS packet_time_sec', f'{_identifier(sub)} AS packet_time_sub']
            records = c.execute(f'SELECT {", ".join(expressions)} FROM {_identifier(table)} WHERE _source_sha256=? ORDER BY packet_index LIMIT {MAX_ROWS+1}', [source_hash]).fetchall()
            if len(records) != row_count:
                raise ValueError(f'APID {apid} ingestion count does not match its table.')
            payload = {'format': 'suncet-packet-v1', 'packet': {k: definition[k] for k in ['apid','name','title']},
                       'title': definition['title'], 'createdUtc': _created_utc(), 'signals': signals,
                       'columns': selected, 'rows': [[_public_value(v) for v in row] for row in records], 'rowCount': len(records)}
            expanded = _json_bytes(payload)
            if len(expanded) > MAX_EXPANDED_BYTES:
                raise ValueError(f'APID {apid} exceeds the expanded snapshot size limit.')
            compressed = gzip.compress(expanded, compresslevel=6, mtime=0)
            if len(compressed) > MAX_COMPRESSED_BYTES:
                raise ValueError(f'APID {apid} exceeds the compressed snapshot size limit.')
            # Content-addressed names let the old catalog work until the new one is uploaded.
            import hashlib
            filename = f'{dataset_id}-apid-{apid:04d}-{hashlib.sha256(compressed).hexdigest()[:12]}.json.gz'
            _atomic_write(output / filename, compressed)
            packets.append({**payload['packet'], 'url': filename, 'rowCount': len(records), 'bytes': len(compressed)})
            print(f"  {definition['title']}: {len(records):,} packets", flush=True)
    return packets
