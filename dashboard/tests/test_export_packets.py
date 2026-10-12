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
                c.execute("INSERT INTO _ingestions VALUES (27,'apid_0027',?,'next',1)",[str(folder/'decoded_packets/decoded_apid_0027_ana_hk__xband_new.csv')])
                c.execute('CREATE TABLE apid_0027 (packet_index INTEGER,ana_cdh_temp DOUBLE,ccsdsSecHeader2_sec_ana_hk BIGINT,ccsdsSecHeader2_sub_ana_hk INTEGER,_source_sha256 VARCHAR,secret VARCHAR)')
                c.execute("INSERT INTO apid_0027 VALUES (1,21,1,27,'selected','private'),(2,22,844672030,572,'selected','private'),(1,99,844672030,572,'other','private'),(1,23,844672040,572,'next','private')")
            before=db.read_bytes()
            packets=export_packets(db,folder,output,'test')
            self.assertEqual(len(packets),1);self.assertEqual(packets[0]['apid'],27)
            payload=json.loads(gzip.decompress((output/packets[0]['url']).read_bytes()))
            self.assertEqual(payload['rowCount'],3)
            rows=[dict(zip(payload['columns'],r)) for r in payload['rows']]
            self.assertEqual([r['ana_cdh_temp'] for r in rows],[21,22,23])
            self.assertEqual([r['packet_index'] for r in rows],[0,1,2])
            self.assertEqual(rows[0]['packet_time_sec'],1)
            self.assertEqual(rows[1]['packet_time_sub'],572)
            self.assertNotIn('secret',payload['columns']);self.assertNotIn('_source_sha256',payload['columns'])
            self.assertEqual(db.read_bytes(),before)

    def test_definitions_exclude_command_fault_and_sequence_packets_and_fields(self):
        definitions=json.loads(DEFINITIONS.read_text())
        self.assertFalse({5,7,12,42,524} & {d['apid'] for d in definitions})
        for d in definitions:
            self.assertFalse(d['name'].lower().startswith('adcs_'))
            fields=[s['field'] for s in d['signals']]
            self.assertEqual(len(fields),len(set(fields)))
            for f in fields:
                self.assertNotRegex(f,r'cmd|opcode|arm_state|^fp_|^seq_|^log_params_')

    def test_dsps_alias_times_raw_measurements_and_log_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'capture';output=root/'out';output.mkdir()
            db=root/'source.duckdb'
            with duckdb.connect(str(db)) as c:
                c.execute('CREATE TABLE _ingestions (apid INTEGER,table_name VARCHAR,source_path VARCHAR,source_sha256 VARCHAR,rows_ingested BIGINT)')
                for apid,name in [(35,'dsps_data'),(23,'dsps_log'),(25,'dsps_param'),(29,'dsps_time')]:
                    c.execute('INSERT INTO _ingestions VALUES (?,?,?,?,1)',[apid,f'apid_{apid:04d}',str(folder/f'decoded_packets/decoded_apid_{apid:04d}_{name}.csv'),name])
                c.execute("CREATE TABLE apid_0035 AS SELECT 0 AS packet_index, 'dsps_data' AS _source_sha256, 844672030 AS ccsdsSecHeader2_sec_dsps_pass, 572 AS ccsdsSecHeader2_sub_dsps_pass, 2281 AS dsps_sensor_board_temp_C, 5072 AS dsps_power_5V_mV, 17 AS dsps_cmd_last_opcode, 999 AS dsps_pass_checksum, 'abcd' AS dsps_data_unknown_tail_2bytes_hex")
                c.execute("CREATE TABLE apid_0023 AS SELECT 0 AS packet_index, 'dsps_log' AS _source_sha256, 3 AS sequence_count, 42 AS dsps_log_message_0")
                c.execute("CREATE TABLE apid_0025 AS SELECT 0 AS packet_index, 'dsps_param' AS _source_sha256, 844672030 AS ccsdsSecHeader2_sec_dsps_parameter_pkt, 573 AS ccsdsSecHeader2_sub_dsps_parameter_pkt, 500 AS dsps_param_3_integration_period")
                c.execute("CREATE TABLE apid_0029 AS SELECT 0 AS packet_index, 'dsps_time' AS _source_sha256, 4000 AS dsps_millisec_since_turn_on")
            packets=export_packets(db,folder,output,'test')
            self.assertEqual({p['apid'] for p in packets},{23,25,29,35})
            payloads={p['apid']:json.loads(gzip.decompress((output/p['url']).read_bytes())) for p in packets}
            data=payloads[35];row=dict(zip(data['columns'],data['rows'][0]))
            self.assertEqual(row['packet_time_sec'],844672030)
            self.assertEqual(row['packet_time_sub'],572)
            self.assertEqual(row['dsps_sensor_board_temp_C'],2281)
            units={s['field']:s['unit'] for s in data['signals']}
            self.assertEqual(units['dsps_sensor_board_temp_C'],'decoded units')
            self.assertEqual(units['dsps_power_5V_mV'],'mV')
            self.assertFalse({'dsps_cmd_last_opcode','dsps_pass_checksum','dsps_data_unknown_tail_2bytes_hex'} & set(data['columns']))
            self.assertEqual(payloads[23]['columns'].count('sequence_count'),1)
            self.assertNotIn('dsps_log_message_0',payloads[23]['columns'])
            param=dict(zip(payloads[25]['columns'],payloads[25]['rows'][0]))
            self.assertEqual(param['packet_time_sub'],573)
