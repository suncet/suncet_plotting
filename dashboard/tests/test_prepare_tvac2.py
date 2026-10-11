"""Check repeatable capture isolation and conservative clock correction."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timezone

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_tvac2 as prep


class TVACPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / '2026-10-07_tvac2'
        self.folder.mkdir()
        self.source = self.root / 'source.duckdb'
        self.catalog = self.root / 'catalog.json'
        self.other = {'id': 'earlier', 'title': 'Earlier', 'url': 'earlier.json.gz', 'rowCount': 2, 'bytes': 100}
        self.catalog.write_text(json.dumps({'format': 'suncet-catalog-v1', 'datasets': [self.other]}))
        raw = self.folder / 'xbandDemod_test.dat'
        raw.write_bytes(b'raw')
        manifests = self.folder / 'processing_manifests'
        manifests.mkdir()
        self.manifest = manifests / 'processing_run_20261010.json'
        self.manifest.write_text(json.dumps({'status': 'succeeded', 'data_root': str(self.folder), 'inputs': [{
            'path': raw.name, 'relative_to_data_root': True, 'size_bytes': 3,
            'modified_utc': datetime.fromtimestamp(raw.stat().st_mtime, timezone.utc).isoformat()}]}))
        with duckdb.connect(str(self.source)) as c:
            c.execute('CREATE TABLE _ingestions (source_path VARCHAR,source_sha256 VARCHAR,rows_ingested BIGINT,apid INTEGER,table_name VARCHAR)')
            c.execute("INSERT INTO _ingestions VALUES (?,?,5,1,'apid_0001')", [str(self.folder / 'decoded_packets/decoded_apid_0001_beacon.csv'), 'tvac'])
            c.execute('CREATE TABLE _apid_catalog (apid INTEGER,packet_name VARCHAR,table_name VARCHAR)')
            c.execute("INSERT INTO _apid_catalog VALUES (1,'beacon','apid_0001')")
            c.execute('CREATE TABLE _database_metadata (key VARCHAR,value VARCHAR)')
            c.execute('''CREATE TABLE apid_0001 (packet_index BIGINT,beac_time_alive BIGINT,
                beac_num_sc_resets INTEGER,ccsdsSecHeader2_sec_beacon BIGINT,
                ccsdsSecHeader2_sub_beacon BIGINT,combined_time_coarse BIGINT,combined_time_fine BIGINT,
                packet_hash VARCHAR,_source_sha256 VARCHAR,beac_ana_eps_bus_v DOUBLE)''')
            # First boot unanchored; second anchored after two samples; forward clock adjustment preserved.
            for index, alive, reset, sec, sub in [(0,100,1,5,27),(1,101,2,1,27),(2,102,2,2,27),
                                                 (3,103,2,844672030,572),(4,104,2,844672060,572)]:
                c.execute('INSERT INTO apid_0001 VALUES (?,?,?,?,?,?,?,?,?,15.5)', [index,alive,reset,sec,sub,sec,sub,str(index),'tvac'])
            c.execute("INSERT INTO apid_0001 VALUES (99,999,9,1,0,1,0,'unrelated','other',99)")

    def tearDown(self):
        self.temp.cleanup()

    def test_repeat_preparation_preserves_clocks_source_and_catalog(self):
        before = prep.sha(self.source)
        report = prep.prepare(self.folder, self.source, self.catalog)
        self.assertEqual((report['beacon_count'],report['backfilled_beacons'],report['utc_unanchored_beacons']), (5,2,1))
        self.assertEqual(prep.sha(self.source), before)
        with duckdb.connect(str(self.folder / prep.OUTPUT_DB), read_only=True) as c:
            self.assertEqual(c.execute('SELECT ccsdsSecHeader2_sec_beacon,ccsdsSecHeader2_sub_beacon FROM apid_0001 WHERE packet_index=3').fetchone(),(844672030,572))
            self.assertEqual(c.execute('SELECT analysis_time_correction_ms FROM apid_0001 WHERE packet_index=4').fetchone(),(0,))
            self.assertEqual(c.execute('SELECT ccsdsSecHeader2_sec_beacon FROM apid_0001 WHERE packet_index=1').fetchone(),(844672028,))
        public = self.folder / 'public_dashboard'
        catalog = json.loads((public/'catalog.json').read_text())
        self.assertEqual(catalog['datasets'][0],self.other)
        payload = json.loads(gzip.decompress((public / '2026-10-07-tvac2.json.gz').read_bytes()))
        self.assertEqual(payload['rowCount'],5)
        self.assertNotIn('_source_sha256',payload['columns'])
        prep.prepare(self.folder,self.source,public/'catalog.json')
        self.assertEqual(len(json.loads((public/'catalog.json').read_text())['datasets']),2)
        self.assertTrue(list((self.folder/'previous_exports').glob('*/public_dashboard/catalog.json')))

    def test_new_raw_data_requires_processing_before_replacing_files(self):
        (self.folder/'xbandDemod_new.dat').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError,'no matching successful'):
            prep.prepare(self.folder,self.source,self.catalog)
        self.assertFalse((self.folder/'public_dashboard').exists())

    def test_unexpected_clock_reset_does_not_replace_outputs(self):
        prep.prepare(self.folder,self.source,self.catalog)
        public = self.folder/'public_dashboard/catalog.json'
        before = public.read_bytes()
        with duckdb.connect(str(self.source)) as c:
            c.execute('UPDATE apid_0001 SET ccsdsSecHeader2_sec_beacon=10 WHERE packet_index=4')
        with self.assertRaisesRegex(ValueError,'returns to an unset clock'):
            prep.prepare(self.folder,self.source,self.catalog)
        self.assertEqual(public.read_bytes(),before)

    def test_regressing_valid_clock_stops(self):
        with duckdb.connect(str(self.source)) as c:
            c.execute('UPDATE apid_0001 SET ccsdsSecHeader2_sec_beacon=844672029 WHERE packet_index=4')
        with self.assertRaisesRegex(ValueError,'UTC goes backwards'):
            prep.prepare(self.folder,self.source,self.catalog)
        self.assertFalse((self.folder/prep.OUTPUT_DB).exists())

    def test_incremental_manifest_and_beacon_batches_are_combined(self):
        raw = self.folder/'xbandDemod_new.dat'
        raw.write_bytes(b'next downlink')
        latest = self.manifest.with_name('processing_run_20261011.json')
        latest.write_text(json.dumps({'status':'succeeded','data_root':str(self.folder),'inputs':[{
            'path':raw.name,'relative_to_data_root':True,'size_bytes':raw.stat().st_size,
            'modified_utc':datetime.fromtimestamp(raw.stat().st_mtime,timezone.utc).isoformat()}]}))
        self.assertEqual(prep.latest_manifest(self.folder),latest.resolve())
        with duckdb.connect(str(self.source)) as c:
            c.execute("INSERT INTO _ingestions VALUES (?, 'new',2,1,'apid_0001')", [str(self.folder/'decoded_packets/decoded_apid_0001_beacon__xband_new.csv')])
            c.execute("INSERT INTO apid_0001 VALUES (0,200,2,844672200,572,844672200,572,'new0','new',15.6),(1,201,2,844672201,572,844672201,572,'new1','new',15.7)")
        report=prep.prepare(self.folder,self.source,self.catalog)
        self.assertEqual(report['beacon_count'],7)
        self.assertEqual(len(report['source_ingestions']),2)
        with duckdb.connect(str(self.folder/prep.OUTPUT_DB),read_only=True) as c:
            self.assertEqual(c.execute('SELECT count(DISTINCT packet_index) FROM apid_0001').fetchone()[0],7)
            self.assertEqual(c.execute("SELECT original_packet_index FROM apid_0001 WHERE packet_hash='new0'").fetchone()[0],0)

    def test_counter_disagreement_is_not_silently_corrected(self):
        with self.assertRaisesRegex(ValueError,'disagree'):
            prep.correction_plan([(0,100,1000,1),(1,101,9000,1),(2,102,844672030572,1)])


if __name__ == '__main__':
    unittest.main()
