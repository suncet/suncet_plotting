import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
import duckdb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_packets import export_packets, DEFINITIONS

class PacketExportTests(unittest.TestCase):
    def test_capture_isolation_allowlist_and_original_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'capture';output=root/'out';output.mkdir()
            db=root/'source.duckdb'
            with duckdb.connect(str(db)) as c:
                c.execute('CREATE TABLE _ingestions (apid INTEGER,table_name VARCHAR,source_path VARCHAR,source_sha256 VARCHAR,rows_ingested BIGINT)')
                c.execute("INSERT INTO _ingestions VALUES (27,'apid_0027',?,'selected',2),(27,'apid_0027',?,'other',1),(12,'apid_0012',?,'private',1)",
                          [str(folder/'decoded_packets/decoded_apid_0027_ana_hk.csv'),str(root/'elsewhere/decoded_packets/a.csv'),str(folder/'decoded_packets/decoded_apid_0012_cmd_hk.csv')])
                c.execute('CREATE TABLE apid_0027 (packet_index INTEGER,ana_cdh_temp DOUBLE,ccsdsSecHeader2_sec_ana_hk BIGINT,ccsdsSecHeader2_sub_ana_hk INTEGER,_source_sha256 VARCHAR,secret VARCHAR)')
                c.execute("INSERT INTO apid_0027 VALUES (1,21,1,27,'selected','private'),(2,22,844672030,572,'selected','private'),(1,99,844672030,572,'other','private')")
            before=db.read_bytes()
            packets=export_packets(db,folder,output,'test')
            self.assertEqual(len(packets),1);self.assertEqual(packets[0]['apid'],27)
            payload=json.loads(gzip.decompress((output/packets[0]['url']).read_bytes()))
            self.assertEqual(payload['rowCount'],2)
            rows=[dict(zip(payload['columns'],r)) for r in payload['rows']]
            self.assertEqual([r['ana_cdh_temp'] for r in rows],[21,22])
            self.assertEqual(rows[0]['packet_time_sec'],1)
            self.assertEqual(rows[1]['packet_time_sub'],572)
            self.assertNotIn('secret',payload['columns']);self.assertNotIn('_source_sha256',payload['columns'])
            self.assertEqual(db.read_bytes(),before)

    def test_definitions_exclude_command_fault_and_sequence_packets_and_fields(self):
        definitions=json.loads(DEFINITIONS.read_text())
        self.assertFalse({5,7,12,42,524} & {d['apid'] for d in definitions})
        for d in definitions:
            fields=[s['field'] for s in d['signals']]
            self.assertEqual(len(fields),len(set(fields)))
            for f in fields:
                self.assertNotRegex(f,r'cmd|opcode|arm_state|^fp_|^seq_|^log_params_')
