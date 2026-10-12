#!/usr/bin/env python3
"""Refresh TVAC2's local upload files after running the telemetry pipeline.

Requires the same Python environment with duckdb used by the processing pipeline.
No data is uploaded. Valid onboard UTC is preserved; only an initial unset clock
with a later anchor in the same boot is backfilled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen

import duckdb

from export_snapshot import export_snapshot, _identifier, MAX_ROWS
from export_packets import export_packets
from capture_batches import capture_selection

DEFAULT_FOLDER = Path(__file__).resolve().parents[3] / 'data/test_data/2026-10-07_tvac2'
CATALOG_URL = 'https://suncet-telemetry.jmason86.workers.dev/catalog.json'
DATASET_ID = '2026-10-07-tvac2'
OUTPUT_DB = '2026-10-07_tvac2_time_corrected.duckdb'
TIME_FIELDS = ['ccsdsSecHeader2_sec_beacon', 'ccsdsSecHeader2_sub_beacon',
               'combined_time_coarse', 'combined_time_fine']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid(ms):
    return ms is not None and 631152000000 <= ms < 3155760000000


def utc(ms):
    # Same FSW epoch/leap-second convention as dashboard/data.mjs (2026).
    return (datetime(2000, 1, 1, tzinfo=timezone.utc) +
            timedelta(milliseconds=ms, seconds=5)).isoformat().replace('+00:00', 'Z')


def correction_plan(rows):
    require(bool(rows), 'The selected TVAC2 ingestion has no beacons.')
    require(len(rows) <= MAX_ROWS, f'TVAC2 exceeds the {MAX_ROWS:,}-row dashboard limit; split the capture before publishing.')
    require(all(all(value is not None for value in row) for row in rows), 'Missing time or reset metadata needs inspection.')
    require(len({r[0] for r in rows}) == len(rows), 'Duplicate packet indices need inspection.')
    require(all(b[1] >= a[1] for a, b in zip(rows, rows[1:])), 'Time alive goes backwards; inspect the ingestion before publishing.')
    boots = defaultdict(list)
    for row in rows:
        boots[row[3]].append(row)
    corrections, unknown = [], []
    for reset, boot in boots.items():
        require(all(r[3] == reset for r in rows if boot[0][0] <= r[0] <= boot[-1][0]),
                'A reset number repeats after another boot; time correction needs inspection.')
        anchors = [r for r in boot if valid(r[2])]
        if not anchors:
            unknown.append({'reset': reset, 'beacons': len(boot)})
            continue
        anchor = anchors[0]
        unset = [r for r in boot if not valid(r[2])]
        require(all(r[0] < anchor[0] for r in unset), f'Reset {reset} returns to an unset clock after a valid time; inspect it first.')
        if unset:
            reference = unset[-1]
            offset = anchor[2] + (reference[1] - anchor[1]) * 1000 - reference[2]
            require(max(abs(r[2] + offset - (anchor[2] + (r[1] - anchor[1]) * 1000)) for r in unset) <= 2,
                    f'Reset {reset}: unset clock and time alive disagree; correction needs inspection.')
            corrections.append({'reset': reset, 'start_packet_index': unset[0][0],
                                'end_packet_index': unset[-1][0], 'offset_ms': offset,
                                'beacons': len(unset), 'anchor_packet_index': anchor[0],
                                'anchor_utc': utc(anchor[2]), 'anchor_time_alive_s': anchor[1]})
    return corrections, unknown


def latest_manifest(folder):
    folder = folder.resolve()
    paths = sorted((folder / 'processing_manifests').glob('processing_run_*.json'))
    require(bool(paths), 'No processing manifest found. Run the processing pipeline first.')
    path = paths[-1]
    manifest = json.loads(path.read_text())
    require(manifest.get('status') == 'succeeded',
            f"Latest processing run is {manifest.get('status', 'unknown')}. Wait for a successful run before preparing uploads.")
    require(Path(manifest['data_root']).resolve() == folder, 'Manifest belongs to a different capture folder.')
    # Incremental runs list only their new input; earlier successful runs cover
    # the existing downlinks. Failed/running manifests never count as coverage.
    inputs = []
    for previous in paths:
        completed = json.loads(previous.read_text())
        if completed.get('status') == 'succeeded' and Path(completed['data_root']).resolve() == folder:
            inputs.extend(i for i in completed['inputs'] if i.get('relative_to_data_root'))
    raw_files = list(folder.glob('xbandDemod*.dat'))
    require(bool(raw_files), 'No TVAC2 X-band input files found.')
    for file in raw_files:
        stat = file.stat()
        candidates = [i for i in inputs if i['size_bytes'] == stat.st_size]
        exact = [i for i in candidates if i['path'] == file.name]
        if any(abs(stat.st_mtime - datetime.fromisoformat(i['modified_utc'].replace('Z', '+00:00')).timestamp()) < 0.01 for i in exact):
            continue
        # Content hashes also recognize renamed files or metadata-only changes.
        if candidates and sha(file) in {i.get('sha256') for i in candidates}:
            continue
        require(False, f'{file.name} has no matching successful processing run. Process this new or changed input first.')
    return path


def build_database(source, folder, dest):
    """Select this capture's ingestion and validate a corrected beacon-only copy."""
    source_hash = sha(source)
    with duckdb.connect(str(source), read_only=True) as c:
        selection, batches = capture_selection(c, folder, 1, beacon_timeline=True)
        require(selection is not None, 'No TVAC2 beacon ingestion found. Run the pipeline first.')
        for path, digest, count, _ in batches:
            csv = Path(path)
            if csv.exists():
                require(sha(csv) == digest, f'Decoded beacon CSV differs from its ingestion: {csv.name}')
        rows = c.execute('SELECT packet_index, beac_time_alive, '
                         'ccsdsSecHeader2_sec_beacon*1000+ccsdsSecHeader2_sub_beacon, beac_num_sc_resets '
                         f'FROM ({selection}) ORDER BY packet_index').fetchall()
        bad = c.execute(f'SELECT count(*) FROM ({selection}) WHERE '
                        '(ccsdsSecHeader2_sub_beacon IS NULL OR ccsdsSecHeader2_sub_beacon NOT BETWEEN 0 AND 999)').fetchone()[0]
        require(not bad, 'Invalid header milliseconds need inspection.')
    corrections, unknown = correction_plan(rows)
    report = {'source_name': source.name, 'source_sha256': source_hash,
              'source_ingestions': [{'path':p,'sha256':h,'rows':n} for p,h,n,_ in batches], 'output_name': dest.name,
              'generated_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Merge beacon playbacks by time alive, retaining original packet indices and overlapping records. Preserve all set clocks. Backfill only initial unset clocks from the first set clock in the same boot using time alive. Never compress offline calendar gaps or extrapolate across boots.',
              'limitation': 'Backfilled times inherit time-setting accuracy and whole-second time-alive precision. Unanchored boots keep their unset clocks and are omitted from UTC plots.',
              'utc_convention': 'Midnight 2000-01-01 plus five post-J2000 leap seconds, matching dashboard/data.mjs.',
              'corrections': corrections, 'unanchored_boots': unknown}
    with duckdb.connect(str(dest)) as c:
        c.execute("ATTACH '" + str(source).replace("'", "''") + "' AS original (READ_ONLY)")
        original_selection, current_batches = capture_selection(c, folder, 1, schema='original', beacon_timeline=True)
        require(current_batches == batches, 'Source ingestions changed during preparation.')
        c.execute(f'CREATE TABLE apid_0001 AS {original_selection}')
        c.execute('CREATE TABLE _apid_catalog AS SELECT * FROM original._apid_catalog WHERE apid=1')
        c.execute('CREATE TABLE _database_metadata AS SELECT * FROM original._database_metadata')
        fields = {r[0]: r[1] for r in c.execute('DESCRIBE apid_0001').fetchall()}
        for field in TIME_FIELDS:
            c.execute(f'ALTER TABLE apid_0001 ADD COLUMN {_identifier("original_"+field)} {fields[field]}')
            c.execute(f'UPDATE apid_0001 SET {_identifier("original_"+field)}={_identifier(field)}')
        c.execute('ALTER TABLE apid_0001 ADD COLUMN analysis_time_correction_ms BIGINT DEFAULT 0')
        c.execute('ALTER TABLE apid_0001 ADD COLUMN analysis_time_reconstructed BOOLEAN DEFAULT false')
        for fix in corrections:
            c.execute('UPDATE apid_0001 SET analysis_time_correction_ms=?, analysis_time_reconstructed=true WHERE packet_index BETWEEN ? AND ?',
                      [fix['offset_ms'], fix['start_packet_index'], fix['end_packet_index']])
        expr = '(original_ccsdsSecHeader2_sec_beacon*1000+original_ccsdsSecHeader2_sub_beacon+analysis_time_correction_ms)'
        c.execute(f'UPDATE apid_0001 SET ccsdsSecHeader2_sec_beacon={expr}//1000, ccsdsSecHeader2_sub_beacon={expr}%1000, combined_time_coarse={expr}//1000, combined_time_fine={expr}%1000 WHERE analysis_time_reconstructed')
        comparisons = [f'a.{_identifier("original_"+f if f in TIME_FIELDS else f)} IS DISTINCT FROM b.{_identifier(f)}' for f in fields]
        changed = c.execute(f'SELECT count(*) FROM apid_0001 a FULL JOIN ({original_selection}) b USING(packet_index) WHERE a.packet_index IS NULL OR b.packet_index IS NULL OR ' + ' OR '.join(comparisons)).fetchone()[0]
        require(not changed, 'Source-field preservation check failed.')
        preserved = ' OR '.join(f'{_identifier(f)} IS DISTINCT FROM {_identifier("original_"+f)}' for f in TIME_FIELDS)
        require(c.execute('SELECT count(*) FROM apid_0001 WHERE NOT analysis_time_reconstructed AND ('+preserved+')').fetchone()[0] == 0,
                'An original set clock changed.')
        corrected = c.execute('SELECT ccsdsSecHeader2_sec_beacon*1000+ccsdsSecHeader2_sub_beacon FROM apid_0001 ORDER BY packet_index').fetchall()
        usable = [r[0] for r in corrected if valid(r[0])]
        require(bool(usable), 'No usable UTC anchors found.')
        require(all(b >= a for a, b in zip(usable, usable[1:])), 'UTC goes backwards. Inspect the clock before publishing; existing upload files are unchanged.')
        prior = folder / OUTPUT_DB
        if prior.exists():
            c.execute("ATTACH '" + str(prior).replace("'", "''") + "' AS prior (READ_ONLY)")
            require(c.execute('SELECT count(*) FROM prior.apid_0001 p WHERE NOT EXISTS (SELECT 1 FROM apid_0001 n WHERE n.packet_hash=p.packet_hash)').fetchone()[0] == 0,
                    'The new ingestion is missing previously prepared packets. Inspect it before replacing the capture.')
            c.execute('DETACH prior')
        report.update(beacon_count=len(rows), original_set_clock_beacons=sum(valid(r[2]) for r in rows),
                      backfilled_beacons=sum(f['beacons'] for f in corrections), utc_usable_beacons=len(usable),
                      utc_unanchored_beacons=len(rows)-len(usable), utc_range=[utc(usable[0]), utc(usable[-1])],
                      validation={'original_fields_preserved': True, 'original_set_clocks_unchanged': True,
                                  'nondecreasing_usable_utc': True})
        c.execute('INSERT INTO _database_metadata VALUES (?,?)', ['analysis_time_reconstruction', json.dumps(report)])
        c.execute('DETACH original')
        c.execute('CHECKPOINT')
    require(sha(source) == source_hash, 'Source database changed during preparation. Wait for processing to finish and retry.')
    report['output_sha256'] = sha(dest)
    report['validation']['source_unchanged'] = True
    return report


def prepare(folder, source, catalog_file=None):
    folder, source = folder.resolve(), source.resolve()
    manifest = latest_manifest(folder)
    manifest_hash = sha(manifest)
    if catalog_file:
        catalog_bytes = catalog_file.read_bytes()
    else:
        # Cloudflare rejects urllib's default Python-urllib user agent (1010).
        request = Request(f'{CATALOG_URL}?refresh={time.time_ns()}', headers={
            'User-Agent': 'SunCET-Telemetry-Preparation/1.0',
            'Accept': 'application/json',
            'Cache-Control': 'no-cache',
        })
        with urlopen(request, timeout=30) as response:
            catalog_bytes = response.read(2 * 1024 * 1024 + 1)
        require(len(catalog_bytes) <= 2 * 1024 * 1024, 'Catalog unexpectedly exceeds 2 MiB.')
    # Stage inside the capture folder: final replacements stay on one filesystem.
    with tempfile.TemporaryDirectory(prefix='.prepare-tvac2-', dir=folder) as temporary:
        stage = Path(temporary)
        public = stage / 'public_dashboard'
        public.mkdir()
        (public / 'catalog.json').write_bytes(catalog_bytes)
        print('Checking TVAC2 clocks and preparing the beacon database...', flush=True)
        report = build_database(source, folder, stage / OUTPUT_DB)
        report.update(processing_manifest=str(manifest), processing_manifest_sha256=manifest_hash)
        audit_name = Path(OUTPUT_DB).with_suffix('.time-correction.json')
        (stage / audit_name).write_text(json.dumps(report, indent=2)+'\n')
        entry = export_snapshot(stage / OUTPUT_DB, public, '2026-10-07 · TVAC2', dataset_id=DATASET_ID)
        print('Preparing additional APID snapshots...', flush=True)
        packets = export_packets(source, folder, public, DATASET_ID)
        catalog = json.loads((public / 'catalog.json').read_text())
        catalog_entry = next(d for d in catalog['datasets'] if d['id'] == DATASET_ID)
        catalog_entry['packets'] = packets
        (public / 'catalog.json').write_text(json.dumps(catalog, separators=(',', ':'))+'\n')
        report['packets'] = [{'apid': p['apid'], 'rowCount': p['rowCount']} for p in packets]
        (stage / audit_name).write_text(json.dumps(report, indent=2)+'\n')
        require(sha(source) == report['source_sha256'], 'Source database changed during preparation. Wait for processing to finish and retry.')
        previous_catalog = json.loads(catalog_bytes)
        published = next((d for d in previous_catalog['datasets'] if d['id'] == DATASET_ID), None)
        require(not published or entry['rowCount'] >= published['rowCount'], 'New capture has fewer rows than the published one. Inspect it before replacing.')
        require(sha(manifest) == manifest_hash and latest_manifest(folder) == manifest,
                'Processing changed during preparation. Wait for it to finish and retry.')
        names = [Path(OUTPUT_DB), audit_name, Path('public_dashboard') / entry['url'],
                 *[Path('public_dashboard') / p['url'] for p in packets], Path('public_dashboard/catalog.json')]
        obsolete = [p.relative_to(folder) for p in (folder / 'public_dashboard').glob(f'{DATASET_ID}-apid-*.json.gz') if p.relative_to(folder) not in names]
        backup = folder / 'previous_exports' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        for name in names + obsolete:
            target = folder / name
            if target.exists():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
                require(sha(saved) == sha(target), 'Backup verification failed.')
        for name in names:
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            expected_hash = sha(stage / name)
            os.replace(stage / name, target)
            require(sha(target) == expected_hash, f'Final file verification failed: {target}')
        for name in obsolete:
            (folder / name).unlink()
    print(f"Ready: {report['beacon_count']:,} beacons; UTC {report['utc_range'][0]} to {report['utc_range'][1]}.")
    print(f"Backfilled {report['backfilled_beacons']:,} early samples; {report['utc_unanchored_beacons']:,} unanchored samples remain available outside the UTC view.")
    print(f"Prepared {len(packets)+1} APID types. Upload all .json.gz files in this folder to the R2 bucket root, then overwrite catalog.json LAST:")
    print(folder / 'public_dashboard')
    print(folder / 'public_dashboard/catalog.json')
    print('Preparation only: nothing has been uploaded.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=DEFAULT_FOLDER)
    parser.add_argument('--database', type=Path, help='Default: shared telemetry database beside the capture folder')
    parser.add_argument('--catalog-file', type=Path, help='Use an explicitly supplied catalog instead of fetching the live catalog')
    args = parser.parse_args()
    source = args.database or args.folder.parent / 'suncet_telemetry_test_phase_v2.0.1.duckdb'
    try:
        prepare(args.folder, source, args.catalog_file)
    except (ValueError, OSError, duckdb.Error, KeyError) as error:
        parser.exit(1, f'Preparation failed: {error}\n')


if __name__ == '__main__':
    main()
