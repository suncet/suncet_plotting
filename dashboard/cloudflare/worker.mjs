// Paste this entire module into a Cloudflare Worker, then add an R2 binding
// named TELEMETRY_BUCKET pointing to the dedicated public-snapshot bucket.
const ALLOWED_ORIGINS = new Set([
  'https://suncet.github.io',
  'http://127.0.0.1:8051',
  'http://localhost:8051',
]);
const SNAPSHOT_PATH = /^\/[A-Za-z0-9][A-Za-z0-9_-]{0,127}\.json\.gz$/;
const ALLOWED_METHODS = 'GET, HEAD, OPTIONS';

function responseHeaders(origin) {
  const headers = new Headers({
    'Vary': 'Origin',
    'X-Content-Type-Options': 'nosniff',
    'Cache-Control': 'no-store',
  });
  if (ALLOWED_ORIGINS.has(origin)) {
    headers.set('Access-Control-Allow-Origin', origin);
    headers.set('Access-Control-Expose-Headers', 'ETag');
  }
  return headers;
}

function unchanged(request, etag) {
  return (request.headers.get('If-None-Match') || '').split(',')
    .some(tag => tag.trim() === '*' || tag.trim().replace(/^W\//, '') === etag);
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin');
    const headers = responseHeaders(origin);
    const reply = (text, status) => {
      headers.set('Content-Type', 'text/plain; charset=utf-8');
      headers.set('Cache-Control', 'no-store');
      headers.delete('Content-Length');
      headers.delete('ETag');
      return new Response(request.method === 'HEAD' ? null : text, { status, headers });
    };

    // These files are public. CORS controls browser access, not authentication.
    if (origin && !ALLOWED_ORIGINS.has(origin)) return reply('Origin not allowed.', 403);
    if (!['GET', 'HEAD', 'OPTIONS'].includes(request.method)) {
      headers.set('Allow', ALLOWED_METHODS);
      return reply('This endpoint is read-only.', 405);
    }

    const path = new URL(request.url).pathname;
    if (path !== '/' && path !== '/catalog.json' && !SNAPSHOT_PATH.test(path)) {
      return reply('Not found.', 404);
    }

    if (request.method === 'OPTIONS') {
      const method = request.headers.get('Access-Control-Request-Method');
      const requestedHeaders = (request.headers.get('Access-Control-Request-Headers') || '')
        .split(',').map(header => header.trim().toLowerCase()).filter(Boolean);
      if (!origin || !['GET', 'HEAD'].includes(method) || requestedHeaders.some(header => header !== 'if-none-match')) {
        return reply('Preflight not allowed.', 403);
      }
      headers.set('Access-Control-Allow-Methods', 'GET, HEAD');
      headers.set('Access-Control-Allow-Headers', 'If-None-Match');
      headers.set('Access-Control-Max-Age', '3600');
      return new Response(null, { status: 204, headers });
    }

    if (path === '/') {
      headers.set('Location', '/catalog.json');
      return new Response(null, { status: 302, headers });
    }
    if (!env.TELEMETRY_BUCKET) {
      return reply('Add an R2 bucket binding named TELEMETRY_BUCKET to this Worker, then deploy.', 503);
    }

    try {
      const key = path.slice(1);
      const object = request.method === 'HEAD'
        ? await env.TELEMETRY_BUCKET.head(key)
        : await env.TELEMETRY_BUCKET.get(key);
      if (!object) return reply('File not found. Upload the exported files at the bucket root.', 404);

      const catalog = key === 'catalog.json';
      // Set known headers instead of forwarding upload metadata. In particular,
      // gzip files remain bytes, with no Content-Encoding or double compression.
      headers.set('Content-Type', catalog ? 'application/json; charset=utf-8' : 'application/gzip');
      headers.set('Cache-Control', catalog
        ? 'public, max-age=0, must-revalidate, no-transform'
        : 'public, max-age=60, must-revalidate, no-transform');
      headers.set('ETag', object.httpEtag);
      if (unchanged(request, object.httpEtag)) {
        if (object.body) await object.body.cancel();
        return new Response(null, { status: 304, headers });
      }
      headers.set('Content-Length', String(object.size));
      return new Response(request.method === 'HEAD' ? null : object.body, { headers });
    } catch {
      return reply('Could not read telemetry from R2. Please try again.', 503);
    }
  },
};
