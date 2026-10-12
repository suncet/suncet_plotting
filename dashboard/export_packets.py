"""Export explicitly allowlisted engineering packets, scoped to one capture.

Keep original spacecraft times. Unset times remain visible on packet order,
and are omitted on UTC. No beacon time correction is borrowed across packets.
"""
import gzip
import json
from pathlib import Path

import duckdb
from capture_batches import capture_selection
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
        for definition in definitions:
            if definition['name'].lower().startswith('adcs_'):
                continue
            selection, matches = capture_selection(c, capture, definition['apid'])
            if selection is None:
                continue
            apid = definition['apid']
            table = matches[0][3]
            row_count = sum(r[2] for r in matches)
            if not row_count:
                continue
            if row_count > MAX_ROWS:
                raise ValueError(f'APID {apid} exceeds {MAX_ROWS:,} rows. Split the capture before publishing.')
            fields = {r[0] for r in c.execute(f'DESCRIBE {_identifier(table)}').fetchall()}
            signals = [s for s in definition['signals'] if s['field'] in fields]
            if not signals:
                continue
            # APID 35 is named dsps_data by processing, but its generated
            # decoder uses dsps_pass; parameters have a similar alias.
            time_names = definition.get('timeHeaderNames', [definition['name']])
            time_fields = next(((f'ccsdsSecHeader2_sec_{name}', f'ccsdsSecHeader2_sub_{name}')
                                for name in time_names
                                if {f'ccsdsSecHeader2_sec_{name}', f'ccsdsSecHeader2_sub_{name}'} <= fields), None)
            # CSIE codegen keeps the secondary header as SHCOARSE/SHFINE.
            explicit_time_fields = definition.get('timeHeaderFields')
            if time_fields is None and explicit_time_fields and set(explicit_time_fields) <= fields:
                time_fields = explicit_time_fields
            selected = list(dict.fromkeys([f for f in CONTEXT if f in fields] + [s['field'] for s in signals]))
            expressions = [_identifier(f) for f in selected]
            if time_fields:
                sec, sub = time_fields
                selected += ['packet_time_sec', 'packet_time_sub']
                expressions += [f'{_identifier(sec)} AS packet_time_sec', f'{_identifier(sub)} AS packet_time_sub']
            records = c.execute(f'SELECT {", ".join(expressions)} FROM ({selection}) ORDER BY packet_index LIMIT {MAX_ROWS+1}').fetchall()
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
