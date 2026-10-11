import test from 'node:test';
import assert from 'node:assert/strict';
import {parseSnapshot,timeEntries} from '../dashboard/data.mjs';
const sample=()=>({format:'suncet-packet-v1',packet:{apid:27,name:'ana_hk',title:'Analog measurements'},
  signals:[{field:'ana_cdh_temp',label:'CDH temperature',kind:'number',unit:'°C',group:'temperatures'}],
  columns:['packet_index','ana_cdh_temp','packet_time_sec','packet_time_sub'],
  rows:[[1,21,3,0],[2,22,844672030,572]],rowCount:2});
test('APID snapshots keep their own telemetry and UTC; unset clocks remain on packet order',()=>{
  const data=parseSnapshot(sample());
  assert.equal(data.packet.apid,27);assert.equal(data.signals[0].field,'ana_cdh_temp');
  const entries=data.rows.map((row,index)=>({row,index}));
  assert.equal(timeEntries(entries,{axis:'record'}).length,2);
  const utc=timeEntries(entries,{axis:'utc'});assert.equal(utc.length,1);
  assert.equal(new Date(utc[0].x).toISOString(),'2026-10-07T07:07:15.572Z');
});
test('packet metadata rejects missing fields, duplicate definitions and invalid APIDs',()=>{
  for(const change of [p=>p.packet.apid=2048,p=>p.signals[0].field='not_exported',p=>p.signals.push(p.signals[0]),p=>p.signals[0].enums={0:42}]){
    const p=sample();change(p);assert.throws(()=>parseSnapshot(p),/Invalid packet/);
  }
});
