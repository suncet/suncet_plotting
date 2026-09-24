import test from 'node:test';
import assert from 'node:assert/strict';
import { SIGNALS, POWER_PAIRS } from '../dashboard/signals.mjs';
import { powerValue } from '../dashboard/data.mjs';
import { engineeringBounds, derivePowerSignal, powerRangeValue, selectRange, panelGroups } from '../dashboard/plot-policy.mjs';

const signal = field => SIGNALS.find(s => s.field === field);
const contains = (range, value) => range[0] <= value && value <= range[1];

test('engineering temperature range fits readings within the requested display limits', () => {
  const temperature = signal('beac_ana_cdh_temp');
  const values = Object.freeze([20, 21, 22, 45, 2e6, -3e6, null]);
  const result = selectRange([temperature], values);
  assert.equal(result.basis, 'engineering');
  assert.ok(contains(result.range, 45));
  assert.ok(result.range[0] >= -20 && result.range[1] <= 50);
  assert.equal(result.outside, 2);
  assert.deepEqual(values, [20, 21, 22, 45, 2e6, -3e6, null]);
  assert.deepEqual(selectRange([temperature], values, 'full'), { range: null, outside: 0, basis: 'full' });
});

test('ordinary readings fit tightly inside broad guards and constant sensors retain useful spans', () => {
  const current = signal('beac_ana_eps_bus_i');
  const result = selectRange([current], [0.39, 0.4, 0.42, 1e9]);
  assert.ok(result.range[1] - result.range[0] >= 0.099999);
  assert.ok(result.range[1] - result.range[0] < 0.2);
  assert.equal(result.outside, 1);
  const temperature = selectRange([signal('beac_ana_cdh_temp')], [20, 20]);
  assert.deepEqual(temperature.range, [17.5, 22.5]);
  assert.deepEqual(selectRange([{ unit: 'V', displayRange: [0, 5] }], [0]).range, [0, 0.5]);
});

test('all-invalid sensor values use known guards instead of auto-scaling to corrupt values', () => {
  const temperature = signal('beac_ana_cdh_temp');
  const result = selectRange([temperature], [1e7, -1e7, 'bad', null]);
  assert.deepEqual(result.range, engineeringBounds(temperature));
  assert.equal(result.outside, 2);
  assert.deepEqual(selectRange([temperature], []), { range: [-20, 50], outside: 0, basis: 'engineering' });
});

test('solar arrays use their wider requested temperature limits', () => {
  for (const temperature of SIGNALS.filter(s => s.group === 'temperatures')) {
    const bounds = temperature.field.includes('_sa_') ? [-50, 100] : [-20, 50];
    assert.deepEqual(engineeringBounds(temperature), bounds);
    const result = selectRange([temperature], [bounds[0], bounds[1], bounds[0] - 1, bounds[1] + 1]);
    assert.deepEqual(result.range, bounds);
    assert.equal(result.outside, 2);
  }
});

test('rail guards include off states and negative noise without adopting Hydra alarm limits', () => {
  assert.deepEqual(engineeringBounds(signal('beac_ana_cdh_3p3_ref')), [-0.25, 5]);
  assert.deepEqual(engineeringBounds(signal('beac_ana_csie_volt')), [-0.5, 15]);
  assert.deepEqual(engineeringBounds(signal('beac_ana_eps_bus_v')), [-1, 32]);
  assert.ok(contains(engineeringBounds(signal('beac_ana_dsps_curr')), 1));
  assert.ok(contains(engineeringBounds(signal('beac_adcs_att_ctrl_sun_point_ang_err')), 181));
  assert.ok(contains(engineeringBounds(signal('beac_dsps_flare_magnitude')), 0));
});

test('derived power uses the rail-specific voltage and current guard extrema', () => {
  for (const pair of POWER_PAIRS) {
    const power = derivePowerSignal(pair);
    const volts = engineeringBounds(signal(pair.voltage)), amps = engineeringBounds(signal(pair.current));
    assert.equal(power.field, `${pair.id}_power`);
    assert.equal(power.unit, 'W');
    for (const v of volts) for (const i of amps) assert.ok(contains(engineeringBounds(power), v * i));
    const fitted = selectRange([power], [5, 6, 1e8]);
    assert.ok(fitted.range[1] < 10);
    assert.equal(fitted.outside, 1);
  }
});

test('power-axis fitting checks source guards even when a corrupt measurement produces plausible wattage', () => {
  const pair = POWER_PAIRS.find(p => p.id === 'eps_bus');
  const metadata = derivePowerSignal(pair);
  const corruptVoltage = Object.freeze({ [pair.voltage]: 305, [pair.current]: 0.4 });
  assert.ok(contains(engineeringBounds(metadata), powerValue(corruptVoltage, pair)));
  assert.equal(powerRangeValue(corruptVoltage, pair), null);
  assert.equal(metadata.rangeValue(corruptVoltage), null);
  assert.equal(powerValue(corruptVoltage, pair), 122);
  assert.equal(powerRangeValue({ [pair.voltage]: 0.1, [pair.current]: 100 }, pair), null);
  assert.equal(powerRangeValue({ [pair.voltage]: 15, [pair.current]: 0.4 }, pair), 6);
  assert.equal(powerRangeValue({ [pair.voltage]: 0, [pair.current]: 0.4 }, pair), 0);
  assert.equal(powerRangeValue({ [pair.voltage]: 15, [pair.current]: 0 }, pair), 0);
  assert.equal(powerRangeValue({ [pair.voltage]: 15, [pair.current]: null }, pair), null);
});

test('counter, elapsed-time and pointer scales reject gross short-selection spikes without fixed mission maxima', () => {
  for (const field of ['sequence_count', 'beac_time_since_boot', 'sw_store_partition_write_hk_beac']) {
    const metadata = signal(field);
    assert.equal(engineeringBounds(metadata), null);
    const result = selectRange([metadata], [10, 11, 12, 1e9, -1e8]);
    assert.equal(result.basis, 'robust');
    assert.ok(result.range[0] >= 0 && result.range[1] < 100);
    assert.equal(result.outside, 2);
    const high = selectRange([metadata], [1e7, 1e7 + 1, 1e7 + 2]);
    assert.ok(contains(high.range, 1e7 + 2));
  }
  const minimal = selectRange([signal('sequence_count')], [10, 11, 1e9]);
  assert.equal(minimal.outside, 1);
  assert.ok(minimal.range[1] < 100);
});

test('counter scaling handles zeros, small changes, empty sets and wholly negative corruption', () => {
  const count = signal('beac_num_sc_resets');
  for (const values of [[0], [0, 0, 0], [0, 0, 1], [9, 9, 10]]) {
    const result = selectRange([count], values);
    assert.ok(result.range[0] >= 0);
    assert.equal(result.outside, 0);
  }
  assert.deepEqual(selectRange([count], [null, '', NaN, Infinity]), { range: null, outside: 0, basis: 'robust' });
  assert.deepEqual(selectRange([count], [-20, -10]), { range: [0, 1], outside: 2, basis: 'robust' });
});

test('typical remains an explicit robust alternative to engineering and full scales', () => {
  const temperature = signal('beac_ana_cdh_temp');
  const values = [0, 1, 2, 45];
  assert.equal(selectRange([temperature], values).outside, 0);
  const result = selectRange([temperature], values, 'typical');
  assert.equal(result.basis, 'typical');
  assert.ok(result.range[1] < 10);
  assert.equal(result.outside, 1);
});

test('every curated numeric family receives a sensible finite scale', () => {
  for (const metadata of SIGNALS.filter(s => s.kind === 'number')) {
    const range = selectRange([metadata], [0, 0, 0]).range;
    assert.ok(range.every(Number.isFinite), metadata.field);
    assert.ok(range[0] < range[1], metadata.field);
  }
  assert.equal(engineeringBounds(signal('beac_mode_system_mode')), null);
  const custom = Object.freeze([0, 12]);
  assert.deepEqual(engineeringBounds({ unit: 'V', displayRange: custom }), [0, 12]);
  assert.notEqual(engineeringBounds({ displayRange: custom }), custom);
  assert.deepEqual(engineeringBounds({ unit: 'V', displayRange: [10, 0] }), [-1, 32]);
});

test('dedicated tabs separate every temperature and electrical reference while retaining all fields', () => {
  const numbers = SIGNALS.filter(s => s.kind === 'number');
  const groups = panelGroups(numbers);
  const fields = groups.flatMap(group => group.signals.map(s => s.field));
  assert.deepEqual(fields.slice().sort(), numbers.map(s => s.field).sort());
  for (const group of groups.filter(g => g.signals.some(s => ['temperatures', 'electrical', 'timing'].includes(s.group)))) {
    assert.equal(group.signals.length, 1, group.title);
  }
  assert.equal(new Set(fields).size, numbers.length);
});

test('overlays only compare related axes, read/write pairs and histogram bins', () => {
  const groups = panelGroups(SIGNALS.filter(s => s.kind === 'number'));
  assert.equal(groups.find(g => g.title === 'Body rates').signals.length, 3);
  assert.equal(groups.find(g => g.title === 'Reaction wheel speeds').signals.length, 3);
  assert.equal(groups.find(g => g.title === 'Housekeeping storage').signals.length, 2);
  assert.equal(groups.find(g => g.title === 'CSIE histogram').signals.length, 6);
  assert.equal(groups.find(g => g.title === 'Visible SPS Sun position').signals.length, 2);
  assert.equal(groups.find(g => g.title === 'X-ray SPS Sun position').signals.length, 2);
  assert.equal(groups.find(g => g.signals[0].field === 'beac_store_partition_write_log').signals.length, 1);
  assert.equal(groups.find(g => g.signals[0].field === 'beac_dsps_flare_level').signals.length, 1);
  const one = signal('beac_adcs_body_rt2');
  assert.deepEqual(panelGroups([one]), [{ title: one.label, signals: [one] }]);
});

test('body-rate display conversion preserves missing values and raw source radians', () => {
  for (const rate of SIGNALS.filter(s => /^beac_adcs_body_rt[123]$/.test(s.field))) {
    assert.equal(rate.unit, 'deg/s');
    const row = Object.freeze({ [rate.field]: Math.PI / 2 });
    assert.equal(rate.derive(row), 90);
    assert.equal(row[rate.field], Math.PI / 2);
    assert.equal(rate.derive({ [rate.field]: -Math.PI }), -180);
    assert.equal(rate.derive({ [rate.field]: 0 }), 0);
    assert.equal(rate.derive({ [rate.field]: '0.5' }), 0.5 * 180 / Math.PI);
    for (const missing of [null, undefined, '', ' ', NaN, Infinity, 'bad', [], {}]) {
      assert.equal(rate.derive({ [rate.field]: missing }), null);
    }
    assert.deepEqual(engineeringBounds(rate), [-180 / Math.PI, 180 / Math.PI]);
    const range = selectRange([rate], [0, 0]).range;
    assert.ok(Math.abs(range[1] - range[0] - 0.01 * 180 / Math.PI) < 1e-12);
  }
});
