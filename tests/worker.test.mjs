import test from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import worker from '../dashboard/cloudflare/worker.mjs';
import { loadSnapshot } from '../dashboard/data.mjs';

const origin = 'https://suncet.github.io';
const snapshot = {
  format: 'suncet-beacon-v1', title: 'Test capture', createdUtc: '2026-09-23T00:00:00Z',
  columns: ['packet_index', 'beac_ana_cdh_temp'], rows: [[0, 25], [1, null]], rowCount: 2,
};
const compressed = gzipSync(JSON.stringify(snapshot));
const catalog = { format: 'suncet-catalog-v1', datasets: [{ id: 'capture', title: 'Test capture', url: 'capture.json.gz' }] };

function fixture() {
  const files = new Map([
    ['catalog.json', Buffer.from(JSON.stringify(catalog))],
    ['capture.json.gz', compressed],
    ['private.duckdb', Buffer.from('not public')],
  ]);
  const calls = [];
  const read = async (key, head) => {
    calls.push([head ? 'head' : 'get', key]);
    const bytes = files.get(key);
    if (!bytes) return null;
    return {
      size: bytes.length, httpEtag: '"v1"',
      // Incorrect upload metadata must not cause gzip decoding twice.
      httpMetadata: { contentEncoding: 'gzip', contentType: 'text/plain' },
      ...(!head ? { body: new Response(bytes).body } : {}),
    };
  };
  const env = { TELEMETRY_BUCKET: { get: key => read(key, false), head: key => read(key, true) } };
  const request = (path, init = {}) => worker.fetch(new Request(`https://telemetry.example${path}`, init), env);
  return { request, env, calls };
}

test('public catalog works from Pages and a direct browser visit', async () => {
  const { request } = fixture();
  const response = await request('/catalog.json', { headers: { Origin: origin } });
  assert.deepEqual(await response.json(), catalog);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), origin);
  assert.equal(response.headers.get('Vary'), 'Origin');
  assert.match(response.headers.get('Cache-Control'), /max-age=0/);
  assert.equal(response.headers.get('Content-Encoding'), null);
  assert.equal((await request('/catalog.json')).status, 200);
  const root = await request('/');
  assert.equal(root.status, 302);
  assert.equal(root.headers.get('Location'), '/catalog.json');
});

test('gzip response preserves exact bytes and loads through the actual dashboard decoder', async t => {
  const { request } = fixture();
  const response = await request('/capture.json.gz', { headers: { Origin: origin } });
  assert.equal(response.headers.get('Content-Type'), 'application/gzip');
  assert.equal(response.headers.get('Content-Encoding'), null);
  assert.equal(Number(response.headers.get('Content-Length')), compressed.length);
  assert.deepEqual(Buffer.from(await response.arrayBuffer()), compressed);
  t.mock.method(globalThis, 'fetch', async () => request('/capture.json.gz'));
  const loaded = await loadSnapshot('https://telemetry.example/capture.json.gz');
  assert.equal(loaded.rows.length, 2);
  assert.equal(loaded.rows[0].beac_ana_cdh_temp, 25);
  assert.equal(loaded.rows[1].beac_ana_cdh_temp, null);
});

test('HEAD reads metadata only and conditional requests return correct empty responses', async () => {
  const { request, calls } = fixture();
  const head = await request('/capture.json.gz', { method: 'HEAD' });
  assert.deepEqual(calls, [['head', 'capture.json.gz']]);
  assert.equal(head.headers.get('ETag'), '"v1"');
  assert.equal(head.headers.get('Content-Length'), String(compressed.length));
  assert.equal(await head.text(), '');
  for (const tag of ['"v1"', 'W/"v1"', '"old", "v1"', '*']) {
    const cached = await request('/capture.json.gz', { headers: { 'If-None-Match': tag, Origin: origin } });
    assert.equal(cached.status, 304);
    assert.equal(await cached.text(), '');
    assert.equal(cached.headers.get('Access-Control-Allow-Origin'), origin);
  }
  assert.equal((await request('/capture.json.gz', { headers: { 'If-None-Match': '"old"' } })).status, 200);
});

test('unsupported paths and write operations never reach R2', async () => {
  const { request, calls } = fixture();
  for (const path of ['/private.duckdb', '/secret.json', '/folder/capture.json.gz', '/%63atalog.json', '/capture.json.gz/extra']) {
    assert.equal((await request(path)).status, 404);
  }
  for (const method of ['PUT', 'POST', 'DELETE', 'PATCH']) {
    const response = await request('/capture.json.gz', { method });
    assert.equal(response.status, 405);
    assert.equal(response.headers.get('Allow'), 'GET, HEAD, OPTIONS');
  }
  assert.deepEqual(calls, []);
});

test('CORS allows Pages and localhost reads but rejects other origins and write preflights', async () => {
  const { request, calls } = fixture();
  const preflight = (site, method = 'GET', headers = 'If-None-Match') => request('/catalog.json', {
    method: 'OPTIONS', headers: { Origin: site, 'Access-Control-Request-Method': method, 'Access-Control-Request-Headers': headers },
  });
  for (const site of [origin, 'http://127.0.0.1:8051', 'http://localhost:8051']) {
    const response = await preflight(site);
    assert.equal(response.status, 204);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), site);
    assert.equal(response.headers.get('Access-Control-Allow-Methods'), 'GET, HEAD');
  }
  assert.equal((await preflight(origin, 'PUT')).status, 403);
  assert.equal((await preflight(origin, 'GET', 'X-Unknown')).status, 403);
  assert.equal((await request('/catalog.json', { headers: { Origin: 'https://other.example' } })).status, 403);
  assert.deepEqual(calls, []);
});

test('missing files, missing bindings and storage errors remain uncached and useful', async () => {
  const { request } = fixture();
  const missing = await request('/missing.json.gz');
  assert.equal(missing.status, 404);
  assert.equal(missing.headers.get('Cache-Control'), 'no-store');
  const unbound = await worker.fetch(new Request('https://telemetry.example/catalog.json'), {});
  assert.equal(unbound.status, 503);
  assert.match(await unbound.text(), /TELEMETRY_BUCKET/);
  const failed = await worker.fetch(new Request('https://telemetry.example/catalog.json'), {
    TELEMETRY_BUCKET: { get() { throw new Error('private implementation detail'); } },
  });
  assert.equal(failed.status, 503);
  assert.equal(failed.headers.get('Cache-Control'), 'no-store');
  assert.doesNotMatch(await failed.text(), /private implementation detail/);
});
