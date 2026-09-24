"""Public exports preserve beacon samples and never include provenance fields."""

import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb


MODULE_PATH = Path(__file__).resolve().parents[1] / "export_snapshot.py"
SPEC = importlib.util.spec_from_file_location("export_snapshot", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.database = self.root / "capture.duckdb"
        self.output = self.root / "published"
        with duckdb.connect(str(self.database)) as connection:
            connection.execute("CREATE TABLE _apid_catalog (apid INTEGER, packet_name VARCHAR, table_name VARCHAR)")
            connection.execute("INSERT INTO _apid_catalog VALUES (1, 'beacon', 'apid_0001')")
            connection.execute("""CREATE TABLE apid_0001 (
                packet_index BIGINT, sequence_count BIGINT,
                beac_num_sc_resets INTEGER, beac_time_since_boot BIGINT,
                beac_ana_eps_bus_v DOUBLE, beac_ana_eps_bus_i DOUBLE,
                beac_mode_system_mode VARCHAR, decode_status VARCHAR,
                is_duplicate_packet BOOLEAN, source_root VARCHAR,
                packet_hash VARCHAR, _ingestion_id UUID, arbitrary_payload INTEGER[]
            )""")
            connection.executemany("INSERT INTO apid_0001 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                (8, 12, 2, 6, float('inf'), 0.2, 'SAFE', 'decoded', False,
                 '/private/secret-source', 'private-hash', '00000000-0000-0000-0000-000000000001', [1, 2]),
                (1, 11, 1, 5, 15.5, None, 'SCIENCE', 'decoded', False,
                 '/private/secret-source', 'private-hash', '00000000-0000-0000-0000-000000000001', [1, 2]),
                (9, 13, 2, 7, float('nan'), float('-inf'), None, 'decoded', True,
                 '/private/secret-source', 'private-hash', '00000000-0000-0000-0000-000000000001', [1, 2]),
            ])

    def tearDown(self):
        self.directory.cleanup()

    def read_snapshot(self, entry):
        compressed = (self.output / entry['url']).read_bytes()
        self.assertEqual(compressed[4:8], b'\x00\x00\x00\x00')
        return json.loads(gzip.decompress(compressed))

    def test_allowlist_missing_values_order_and_source_unchanged(self):
        original = self.database.read_bytes()
        entry = exporter.export_snapshot(self.database, self.output, 'Flight capture', dataset_id='flight')
        snapshot = self.read_snapshot(entry)
        self.assertEqual(snapshot['format'], 'suncet-beacon-v1')
        self.assertEqual(snapshot['rowCount'], 3)
        self.assertTrue(snapshot['createdUtc'].endswith('Z'))
        rows = [dict(zip(snapshot['columns'], row)) for row in snapshot['rows']]
        self.assertEqual([row['packet_index'] for row in rows], [1, 8, 9])
        self.assertEqual(rows[0]['beac_ana_eps_bus_v'], 15.5)
        self.assertIsNone(rows[0]['beac_ana_eps_bus_i'])
        self.assertIsNone(rows[1]['beac_ana_eps_bus_v'])
        self.assertIsNone(rows[2]['beac_ana_eps_bus_v'])
        self.assertIsNone(rows[2]['beac_ana_eps_bus_i'])
        self.assertIsNone(rows[2]['beac_mode_system_mode'])
        self.assertIs(rows[2]['is_duplicate_packet'], True)
        for secret in ['source_root', 'private-hash', 'secret-source', '_ingestion_id', 'arbitrary_payload']:
            self.assertNotIn(secret, json.dumps(snapshot))
        self.assertEqual(self.database.read_bytes(), original)
        self.assertEqual(entry['bytes'], (self.output / entry['url']).stat().st_size)

    def test_reset_filter_and_catalog_upsert_preserve_other_captures(self):
        first = exporter.export_snapshot(self.database, self.output, 'First', dataset_id='first', reset=1)
        second = exporter.export_snapshot(self.database, self.output, 'Second', dataset_id='second', reset=2)
        updated = exporter.export_snapshot(self.database, self.output, 'Renamed', dataset_id='first', reset=2)
        catalog = json.loads((self.output / 'catalog.json').read_text())
        self.assertEqual(catalog, {'format': 'suncet-catalog-v1', 'datasets': [updated, second]})
        self.assertEqual(first['rowCount'], 1)
        self.assertEqual(updated['rowCount'], 2)
        snapshot = self.read_snapshot(updated)
        reset_index = snapshot['columns'].index('beac_num_sc_resets')
        self.assertEqual({row[reset_index] for row in snapshot['rows']}, {2})
        self.assertEqual(list(self.output.glob('.*.tmp')), [])

    def test_gzip_is_deterministic_for_identical_snapshot(self):
        with patch.object(exporter, '_created_utc', return_value='2026-09-23T00:00:00Z'):
            entry = exporter.export_snapshot(self.database, self.output, 'Capture')
            original = (self.output / entry['url']).read_bytes()
            exporter.export_snapshot(self.database, self.output, 'Capture')
        self.assertEqual((self.output / entry['url']).read_bytes(), original)

    def test_row_limit_is_applied_after_reset_filter(self):
        with patch.object(exporter, 'MAX_ROWS', 2):
            with self.assertRaisesRegex(ValueError, 'limit'):
                exporter.export_snapshot(self.database, self.output, 'Too large')
            self.assertFalse(self.output.exists())
            entry = exporter.export_snapshot(self.database, self.output, 'Selected reset', reset=2)
            self.assertEqual(entry['rowCount'], 2)

    def test_size_limits_reject_before_writing_or_replacing_captures(self):
        entry = exporter.export_snapshot(self.database, self.output, 'Original', dataset_id='capture')
        snapshot_path = self.output / entry['url']
        catalog_path = self.output / 'catalog.json'
        original_snapshot = snapshot_path.read_bytes()
        original_catalog = catalog_path.read_bytes()
        for constant, error in [
            ('MAX_EXPANDED_BYTES', 'expanded-size limit'),
            ('MAX_COMPRESSED_BYTES', 'download-size limit'),
        ]:
            with self.subTest(constant=constant), patch.object(exporter, constant, 1):
                with self.assertRaisesRegex(ValueError, error):
                    exporter.export_snapshot(self.database, self.output, 'Rejected replacement', dataset_id='capture')
                self.assertEqual(snapshot_path.read_bytes(), original_snapshot)
                self.assertEqual(catalog_path.read_bytes(), original_catalog)
                with self.assertRaisesRegex(ValueError, error):
                    exporter.export_snapshot(self.database, self.output, 'Rejected new capture', dataset_id='new')
                self.assertFalse((self.output / 'new.json.gz').exists())
                self.assertEqual(catalog_path.read_bytes(), original_catalog)
                missing_output = self.root / f'never-created-{constant}'
                with self.assertRaisesRegex(ValueError, error):
                    exporter.export_snapshot(self.database, missing_output, 'Rejected first capture')
                self.assertFalse(missing_output.exists())
                self.assertEqual(list(self.output.glob('.*.tmp')), [])

    def test_unsafe_id_and_empty_selection_do_not_write(self):
        for unsafe_id in ['../outside', '/absolute', 'with spaces', '.hidden', 'a/b']:
            with self.subTest(dataset_id=unsafe_id), self.assertRaises(ValueError):
                exporter.export_snapshot(self.database, self.output, 'Capture', dataset_id=unsafe_id)
        with self.assertRaisesRegex(ValueError, 'No beacon rows'):
            exporter.export_snapshot(self.database, self.output, 'Missing reset', reset=999)
        self.assertFalse(self.output.exists())

    def test_invalid_catalog_is_not_replaced(self):
        self.output.mkdir()
        catalog = self.output / 'catalog.json'
        catalog.write_text('{"format":"something-else","datasets":[]}')
        original = catalog.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Existing catalog'):
            exporter.export_snapshot(self.database, self.output, 'Capture')
        self.assertEqual(catalog.read_bytes(), original)
        self.assertEqual(list(self.output.glob('*.gz')), [])


if __name__ == '__main__':
    unittest.main()
