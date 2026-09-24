import test from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { numericValue, stateLabel, valueOf, powerValue, parseSnapshot, loadSnapshot, quoteIdentifier, sampleSeries } from '../dashboard/data.mjs';

test('missing and invalid values stay missing while real zero survives', () => {
  for (const missing of [null, undefined, '', '  ', NaN, Infinity, -Infinity, 'SAFE', {}, []]) {
    assert.equal(numericValue(missing), null);
  }
  assert.equal(numericValue(0), 0);
  assert.equal(numericValue('0'), 0);
  assert.equal(numericValue(' -12.5 '), -12.5);
  assert.equal(numericValue(123n), 123);
  assert.equal(numericValue(false), 0);
});

test('decoded state strings survive and numeric state codes use explicit enums', () => {
  const signal = { enums: { 0: 'OFF', 1: 'ON' } };
  assert.equal(stateLabel(signal, 'SUN_POINT'), 'SUN_POINT');
  assert.equal(stateLabel(signal, 0), 'OFF');
  assert.equal(stateLabel(signal, '1'), 'ON');
  assert.equal(stateLabel(signal, false), 'OFF');
  assert.equal(stateLabel(signal, 9), '9');
  assert.equal(stateLabel(signal, null), 'No data');
});

test('signal values are read by schema field and derived values retain nulls', () => {
  assert.equal(valueOf({ rail_v: 0 }, { field: 'rail_v' }), 0);
  assert.equal(valueOf({}, { field: 'rail_v' }), undefined);
  const power = { derive: row => {
    const volts = numericValue(row.rail_v);
    const amps = numericValue(row.rail_i);
    return volts === null || amps === null ? null : volts * amps;
  } };
  assert.equal(valueOf({ rail_v: 12, rail_i: 0.25 }, power), 3);
  assert.equal(valueOf({ rail_v: 12, rail_i: null }, power), null);
});

test('database-provided identifiers cannot escape their quoted identifier', () => {
  assert.equal(quoteIdentifier('apid_0001'), '"apid_0001"');
  assert.equal(quoteIdentifier('odd"; DROP TABLE telemetry;--'), '"odd""; DROP TABLE telemetry;--"');
  assert.throws(() => quoteIdentifier(''), /Invalid database identifier/);
  assert.throws(() => quoteIdentifier('bad\0name'), /Invalid database identifier/);
});

test('numeric downsampling retains spikes, endpoints, and chronological order', () => {
  const xs = Array.from({ length: 10000 }, (_, index) => index);
  const ys = xs.map(() => 4);
  ys[403] = 1000;
  ys[404] = -700;
  const sampled = sampleSeries(xs, ys, 100);
  assert.ok(sampled.x.length <= 100);
  assert.equal(sampled.x[0], 0);
  assert.equal(sampled.x.at(-1), 9999);
  assert.equal(sampled.y[sampled.x.indexOf(403)], 1000);
  assert.equal(sampled.y[sampled.x.indexOf(404)], -700);
  assert.ok(sampled.x.every((value, index) => index === 0 || value > sampled.x[index - 1]));
});

test('sampling preserves missing-data boundaries instead of bridging the gap', () => {
  const xs = Array.from({ length: 100 }, (_, index) => index);
  const ys = xs.map(() => 1);
  ys.fill(null, 40, 60);
  const sampled = sampleSeries(xs, ys, 10);
  for (const index of [39, 40, 59, 60]) assert.ok(sampled.x.includes(index));
  assert.equal(sampled.y[sampled.x.indexOf(40)], null);
  assert.equal(sampled.y[sampled.x.indexOf(59)], null);
});

test('state compression retains all transitions and gaps without inventing values', () => {
  const xs = Array.from({ length: 12 }, (_, index) => index);
  const ys = ['SAFE', 'SAFE', 'SAFE', 'SUN_POINT', 'SUN_POINT', null, null, 'SAFE', 'SAFE', 'SAFE', 'SAFE', 'SAFE'];
  const sampled = sampleSeries(xs, ys, 4, 'state');
  assert.deepEqual(sampled.x, [0, 2, 3, 4, 5, 6, 7, 11]);
  assert.deepEqual(sampled.y, ['SAFE', 'SAFE', 'SUN_POINT', 'SUN_POINT', null, null, 'SAFE', 'SAFE']);
  const changing = sampleSeries(xs, xs.map(index => index % 2), 4, 'state');
  assert.deepEqual(changing.x, xs);
  assert.ok(changing.y.every(value => value === 0 || value === 1));
});

test('empty and short numeric series remain faithful; mismatched coordinates fail', () => {
  assert.deepEqual(sampleSeries([], []), { x: [], y: [] });
  assert.deepEqual(sampleSeries([0, 1, 2], [0, undefined, 2]), { x: [0, 1, 2], y: [0, null, 2] });
  assert.throws(() => sampleSeries([1], []), /equal lengths/);
});

test('derived power uses same-packet measurements and preserves zero and missing data', () => {
  const pair = { voltage: 'volts', current: 'amps' };
  assert.equal(powerValue({ volts: 12, amps: 0.25 }, pair), 3);
  assert.equal(powerValue({ volts: 12, amps: 0 }, pair), 0);
  assert.equal(powerValue({ volts: 0, amps: 0.25 }, pair), 0);
  assert.equal(powerValue({ volts: 12, amps: -0.25 }, pair), -3);
  for (const missing of [null, undefined, '', NaN, Infinity]) {
    assert.equal(powerValue({ volts: 12, amps: missing }, pair), null);
    assert.equal(powerValue({ volts: missing, amps: 0.25 }, pair), null);
  }
  assert.equal(powerValue({ volts: 1e308, amps: 1e308 }, pair), null);
});

const snapshot = overrides => ({
  format: 'suncet-beacon-v1',
  title: 'Published flight playback',
  createdUtc: '2026-09-23T12:00:00Z',
  columns: ['packet_index', 'beac_ana_eps_bus_v', 'beac_mode_system_mode', 'is_duplicate_packet'],
  rows: [[0, 0, 'SAFE', false], [1, null, 'SUN_POINT', true]],
  rowCount: 2,
  ...overrides,
});

test('snapshot parsing preserves genuine zeros, missing values, states, and booleans', () => {
  const result = parseSnapshot(snapshot());
  assert.deepEqual(result.rows, [
    { packet_index: 0, beac_ana_eps_bus_v: 0, beac_mode_system_mode: 'SAFE', is_duplicate_packet: false },
    { packet_index: 1, beac_ana_eps_bus_v: null, beac_mode_system_mode: 'SUN_POINT', is_duplicate_packet: true },
  ]);
  assert.equal(result.filename, 'Published flight playback');
  assert.equal(result.dbVersion, 'snapshot');
  assert.deepEqual(result.catalog, [{ apid: 1, packet_name: 'beacon', table_name: 'beacon' }]);
});

test('malformed snapshot schemas and inconsistent rows fail before plotting', () => {
  for (const payload of [null, [], {}, snapshot({ format: 'unknown' })]) {
    assert.throws(() => parseSnapshot(payload), /snapshot format/);
  }
  assert.throws(() => parseSnapshot(snapshot({ columns: [] })), /columns/);
  assert.throws(() => parseSnapshot(snapshot({ columns: ['beac_ana_eps_bus_v', 'beac_ana_eps_bus_v'] })), /duplicate/);
  assert.throws(() => parseSnapshot(snapshot({ columns: ['unknown_column'] })), /no recognized/);
  assert.throws(() => parseSnapshot(snapshot({ rowCount: 3 })), /does not match/);
  assert.throws(() => parseSnapshot(snapshot({ rowCount: -1 })), /nonnegative integer/);
  assert.throws(() => parseSnapshot(snapshot({ rows: [[0]], rowCount: 1 })), /column count/);
  assert.throws(() => parseSnapshot(snapshot({ rows: [], rowCount: 0 })), /no beacon packets/);
  assert.throws(() => parseSnapshot(snapshot({ title: {} })), /title must be text/);
  for (const value of [{ nested: true }, [1], undefined, Infinity, NaN]) {
    assert.throws(() => parseSnapshot(snapshot({ rows: [[0, value, 'SAFE', false]], rowCount: 1 })), /non-scalar or invalid/);
  }
});

test('snapshot row limit checks do not need an oversized allocation', () => {
  assert.throws(() => parseSnapshot(snapshot({ rowCount: 500001 })), /500,000 beacon-row limit/);
  const sparseRows = [];
  sparseRows.length = 500001;
  assert.throws(() => parseSnapshot(snapshot({ rows: sparseRows })), /500,000 beacon-row limit/);
});

test('published snapshot loads ordinary JSON and sets the remote filename fallback', async t => {
  const payload = snapshot({ title: '' });
  const encoded = JSON.stringify(payload);
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, 'https://example.org/flight%20snapshot.json');
    assert.equal(options.credentials, 'omit');
    assert.equal(options.mode, 'cors');
    return new Response(encoded);
  });
  const result = await loadSnapshot('https://example.org/flight%20snapshot.json');
  assert.equal(result.filename, 'flight snapshot.json');
  assert.equal(result.byteSize, Buffer.byteLength(encoded));
  assert.equal(result.rows.length, 2);
});

test('gzip sniffing accepts compressed files and already decoded HTTP gzip responses', async t => {
  const encoded = JSON.stringify(snapshot());
  const compressed = gzipSync(encoded);
  const responses = [
    new Response(compressed),
    new Response(encoded, { headers: { 'Content-Encoding': 'gzip', 'Content-Length': String(compressed.length) } }),
  ];
  t.mock.method(globalThis, 'fetch', async () => responses.shift());
  const fileGzip = await loadSnapshot('https://example.org/snapshot.json.gz');
  const httpGzip = await loadSnapshot('https://example.org/snapshot.json.gz');
  assert.deepEqual(fileGzip.rows, httpGzip.rows);
  assert.equal(fileGzip.byteSize, compressed.length);
  assert.equal(httpGzip.byteSize, compressed.length);
});

test('snapshot loader rejects oversized advertised downloads without allocating their body', async t => {
  const compressed = gzipSync(JSON.stringify(snapshot()));
  t.mock.method(globalThis, 'fetch', async () => new Response(compressed, {
    headers: { 'Content-Length': String(100 * 1024 * 1024 + 1) },
  }));
  await assert.rejects(loadSnapshot('https://example.org/huge.json.gz'), /100 MB download limit/);
});

test('snapshot errors explain invalid URLs, CORS, server failures, and preview pages', async t => {
  await assert.rejects(loadSnapshot('file:///tmp/snapshot.json'), /HTTP or HTTPS/);
  await assert.rejects(loadSnapshot('not a URL'), /URL is invalid/);
  const fetchMock = t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(loadSnapshot('https://example.org/snapshot.json'), /CORS/);
  fetchMock.mock.mockImplementation(async () => new Response('Forbidden', { status: 403 }));
  await assert.rejects(loadSnapshot('https://example.org/snapshot.json'), /HTTP 403/);
  fetchMock.mock.mockImplementation(async () => new Response('<html>Sign in</html>'));
  await assert.rejects(loadSnapshot('https://example.org/snapshot.json'), /sharing preview or sign-in page/);
});
