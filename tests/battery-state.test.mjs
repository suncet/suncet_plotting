import test from 'node:test';
import assert from 'node:assert/strict';
import { batteryChargeState, batteryStateSeries } from '../dashboard/data.mjs';
import { POWER_PAIRS } from '../dashboard/signals.mjs';

test('battery states accept decoded labels and codes without treating missing state as discharging', () => {
  for (const raw of [1, true, '1', 'CHARGING']) assert.equal(batteryChargeState(raw), 'CHARGING');
  for (const raw of [0, false, '0', 'DISCHARGING']) assert.equal(batteryChargeState(raw), 'DISCHARGING');
  for (const raw of [null, undefined, '', 7, 'INVALID']) assert.equal(batteryChargeState(raw), 'UNKNOWN');
});

test('each battery uses its own state and state transitions retain endpoints without joining separated runs', () => {
  const pairs=POWER_PAIRS.filter(p=>p.chargingState);
  const view=[0,1,2,3].map((x,index)=>({x,segment:0,row:{
    beac_batt1_charging_state:[1,0,1,1][index],beac_batt2_charging_state:[0,1,0,0][index],
  }}));
  const first=batteryStateSeries(view,[10,20,30,40],pairs[0].chargingState);
  const second=batteryStateSeries(view,[10,20,30,40],pairs[1].chargingState);
  assert.deepEqual(first.find(s=>s.state==='CHARGING').y,[10,20,null,30,40,null]);
  assert.deepEqual(first.find(s=>s.state==='DISCHARGING').y,[20,30,null]);
  assert.deepEqual(second.find(s=>s.state==='CHARGING').y,[20,30,null]);
});

test('reboots and missing measurements remain gaps, while brief states survive downsampling', () => {
  const view=Array.from({length:200},(_,x)=>({x,segment:x<100?0:1,row:{state:x===50?0:1}}));
  const values=view.map(e=>e.x===80?null:e.x);
  const traces=batteryStateSeries(view,values,'state',8);
  assert.deepEqual(traces.find(s=>s.state==='DISCHARGING').y,[50,51,null]);
  const charging=traces.find(s=>s.state==='CHARGING');
  for(const gap of [80,99])assert(charging.x.some((x,i)=>x===gap&&charging.y[i]===null));
  assert(charging.y.includes(99));
  assert(charging.y.includes(100));
});
