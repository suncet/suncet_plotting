import test from 'node:test';
import assert from 'node:assert/strict';
import {headerSeconds,spacecraftUtcMs,axisValue,axisRangeValue,timeEntries,sampleTelemetry} from '../dashboard/data.mjs';
const row=(sec=843505480,sub=371,alive=716123,reset=232)=>({ccsdsSecHeader2_sec_beacon:sec,ccsdsSecHeader2_sub_beacon:sub,beac_time_alive:alive,beac_time_since_boot:10,beac_num_sc_resets:reset});
test('UTC matches realtime display reference and the CPT clock-setting sample',()=>{
 assert.equal(new Date(spacecraftUtcMs(833237728)).toISOString(),'2026-05-27T22:55:33.000Z');
 assert.equal(new Date(axisValue(row(),'utc')).toISOString(),'2026-09-23T19:04:45.371Z');
 assert.equal(spacecraftUtcMs(0),Date.UTC(2000,0,1));
 assert.equal(axisValue(row(9333,28),'utc'),null);
 assert.equal(axisValue(row(9333,28),'header'),9333.028);
});
test('invalid fine times and missing counters do not invent zero time',()=>{
 for(const v of [null,undefined,'',-1,1000,1.5,Infinity])assert.equal(headerSeconds({...row(100),ccsdsSecHeader2_sub_beacon:v}),null);
 assert.equal(axisValue({},'alive'),null);
 assert.equal(axisValue({...row(),beac_time_alive:0},'alive'),0);
});
test('Plotly UTC zoom strings are parsed independently of local timezone',()=>{
 const expected=Date.parse('2026-09-23T19:04:45.371Z');
 for(const v of ['2026-09-23 19:04:45.371','2026-09-23T19:04:45.371Z',expected])assert.equal(axisRangeValue(v,'utc'),expected);
 assert.equal(axisRangeValue('nonsense','utc'),null);
 assert.equal(axisRangeValue('123.5','alive'),123.5);
});
test('time alive survives reboot, with gaps instead of false connecting lines',()=>{
 const entries=[{index:1,row:row(843505480,371,100,232)},{index:2,row:row(843505481,371,101,232)},{index:3,row:row(10,27,120,235)}];
 const alive=timeEntries(entries,{axis:'alive'});
 assert.deepEqual(alive.map(e=>e.x),[100,101,120]);
 assert.equal(alive[0].segment,alive[1].segment);
 assert.notEqual(alive[1].segment,alive[2].segment);
 assert.deepEqual(sampleTelemetry(alive,[5,6,7]).y,[5,6,null,7]);
 const utc=timeEntries(entries,{axis:'utc'});
 assert.equal(utc.length,2);
 assert.equal(timeEntries(entries,{axis:'alive',reset:'235'}).length,1);
});
test('clock setting within a reset breaks traces and duplicate filtering stays active',()=>{
 const entries=[{index:1,row:row()},{index:2,row:{...row(),is_duplicate_packet:true}},{index:3,row:row(843509000,371,716124)}];
 const view=timeEntries(entries,{axis:'utc'});
 assert.equal(view.length,2);assert.notEqual(view[0].segment,view[1].segment);
 assert.equal(timeEntries(entries,{axis:'utc',hideDuplicates:false}).length,3);
});
