import { SIGNALS } from './signals.mjs';

// Keep the JS API, WASM binary, and worker on the exact same release.
const DUCKDB_PACKAGE = 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.33.1-dev57.0';
const MAX_ROWS = 500_000;
const MAX_COMPRESSED_BYTES = 100 * 1024 * 1024;
const MAX_EXPANDED_BYTES = 400 * 1024 * 1024;
const CONTEXT_FIELDS = [
  'packet_index', 'source_packet_index', 'sequence_count',
  'beac_num_sc_resets', 'beac_time_since_boot',
  'ccsdsSecHeader2_sec_beacon', 'ccsdsSecHeader2_sub_beacon',
  'combined_time_coarse', 'combined_time_fine',
  'decode_status', 'is_duplicate_packet',
];

/** Missing telemetry is not zero. Numeric strings are accepted for older exports. */
export function numericValue(value) {
  if (value === null || value === undefined || typeof value === 'object') return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  if (!['number', 'bigint', 'boolean', 'string'].includes(typeof value)) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

/** Preserve decoded flight-software labels; only translate actual numeric codes. */
export function stateLabel(signal, value) {
  if (value === null || value === undefined || value === '') return 'No data';
  const code = numericValue(value);
  const labels = signal.enums ?? {};
  if (code !== null && Object.hasOwn(labels, String(code))) return String(labels[code]);
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'TRUE' : 'FALSE';
  return code === null ? 'No data' : String(code);
}

/** Read a curated signal, including optional derived channels. */
export function valueOf(row, signal) {
  if (typeof signal.derive === 'function') return signal.derive(row);
  return row[signal.field];
}

/** Voltage and current must come from the same packet; missing is never zero. */
export function powerValue(row, pair) {
  const voltage = numericValue(row[pair.voltage]);
  const current = numericValue(row[pair.current]);
  return voltage === null || current === null ? null : numericValue(voltage * current);
}

/** Validate a compact public snapshot before turning positional rows into records. */
export function parseSnapshot(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload) || payload.format !== 'suncet-beacon-v1') {
    throw new Error('Unsupported snapshot format. Use a suncet-beacon-v1 telemetry export.');
  }
  const { columns, rows, rowCount } = payload;
  if (!Array.isArray(columns) || !columns.length || columns.some(column => typeof column !== 'string' || !column || column.includes('\0'))) {
    throw new Error('Snapshot columns must be a nonempty list of field names.');
  }
  const available = new Set(columns);
  if (available.size !== columns.length) throw new Error('Snapshot contains duplicate column names.');
  if (!SIGNALS.some(signal => available.has(signal.field))) {
    throw new Error('Snapshot contains no recognized SunCET beacon telemetry points.');
  }
  if (!Number.isSafeInteger(rowCount) || rowCount < 0) {
    throw new Error('Snapshot rowCount must be a nonnegative integer.');
  }
  if (rowCount > MAX_ROWS || (Array.isArray(rows) && rows.length > MAX_ROWS)) {
    throw new Error(`Snapshot exceeds the ${MAX_ROWS.toLocaleString()} beacon-row limit. Publish a smaller time range.`);
  }
  if (!Array.isArray(rows) || rows.length !== rowCount) {
    throw new Error('Snapshot rowCount does not match its rows. The export may be incomplete.');
  }
  if (!rows.length) throw new Error('This telemetry snapshot contains no beacon packets.');
  if (payload.title !== undefined && typeof payload.title !== 'string') {
    throw new Error('Snapshot title must be text.');
  }
  if (payload.createdUtc !== undefined && typeof payload.createdUtc !== 'string') {
    throw new Error('Snapshot createdUtc must be text.');
  }
  const records = new Array(rows.length);
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    if (!Array.isArray(row) || row.length !== columns.length) {
      throw new Error(`Snapshot row ${rowIndex + 1} does not match the column count.`);
    }
    const entries = new Array(columns.length);
    for (let columnIndex = 0; columnIndex < columns.length; columnIndex += 1) {
      const value = row[columnIndex];
      const scalar = value === null || typeof value === 'string' || typeof value === 'boolean' ||
        (typeof value === 'number' && Number.isFinite(value));
      if (!scalar) throw new Error(`Snapshot row ${rowIndex + 1}, field ${columns[columnIndex]} contains a non-scalar or invalid value.`);
      entries[columnIndex] = [columns[columnIndex], value];
    }
    records[rowIndex] = Object.fromEntries(entries);
  }
  return {
    rows: records,
    columns: [...columns],
    catalog: [{ apid: 1, packet_name: 'beacon', table_name: 'beacon' }],
    tableName: 'beacon',
    filename: payload.title?.trim() || 'Telemetry snapshot',
    dbVersion: 'snapshot',
  };
}

async function readSnapshotText(response, onProgress) {
  if (!response.body) throw new Error('The snapshot download has no response body.');
  const reader = response.body.getReader();
  const prefix = [];
  let firstBytes = [];
  let transferred = 0;
  let finished = false;
  let streamReader;
  try {
    // Read only enough to distinguish a gzip file from JSON. Content-Encoding
    // may already have been decoded by fetch, so file extensions are unreliable.
    while (firstBytes.length < 2 && !finished) {
      const result = await reader.read();
      finished = result.done;
      if (!result.done) {
        prefix.push(result.value);
        transferred += result.value.byteLength;
        firstBytes = [...firstBytes, ...result.value.subarray(0, 2 - firstBytes.length)];
      }
    }
    const compressed = firstBytes[0] === 0x1f && firstBytes[1] === 0x8b;
    const encodedGzip = /\bgzip\b/i.test(response.headers.get('content-encoding') ?? '');
    const maxTransfer = compressed ? MAX_COMPRESSED_BYTES : MAX_EXPANDED_BYTES;
    const statedSize = Number(response.headers.get('content-length'));
    const maxStatedSize = compressed || encodedGzip ? MAX_COMPRESSED_BYTES : MAX_EXPANDED_BYTES;
    if (statedSize > maxStatedSize || transferred > maxTransfer) {
      throw new Error(`Snapshot exceeds the ${maxStatedSize / 1024 / 1024} MB download limit. Publish a smaller time range.`);
    }
    let reportedMB = -1;
    const source = new ReadableStream({
      start(controller) {
        for (const chunk of prefix) controller.enqueue(chunk);
        if (finished) controller.close();
      },
      async pull(controller) {
        const result = await reader.read();
        if (result.done) { controller.close(); return; }
        transferred += result.value.byteLength;
        if (transferred > maxTransfer) {
          await reader.cancel();
          controller.error(new Error(`Snapshot exceeds the ${maxTransfer / 1024 / 1024} MB download limit. Publish a smaller time range.`));
          return;
        }
        const receivedMB = Math.floor(transferred / 1024 / 1024);
        if (receivedMB !== reportedMB) {
          reportedMB = receivedMB;
          onProgress(`Downloading telemetry snapshot… ${(transferred / 1024 / 1024).toFixed(1)} MB`);
        }
        controller.enqueue(result.value);
      },
      cancel(reason) { return reader.cancel(reason); },
    });
    let decoded = source;
    if (compressed) {
      if (typeof DecompressionStream === 'undefined') {
        await source.cancel();
        throw new Error('This browser cannot decompress gzip snapshots. Use a current browser or publish an uncompressed .json snapshot.');
      }
      decoded = source.pipeThrough(new DecompressionStream('gzip'));
    }
    streamReader = decoded.getReader();
    const textDecoder = new TextDecoder('utf-8', { fatal: true });
    const text = [];
    let expanded = 0;
    while (true) {
      const result = await streamReader.read();
      if (result.done) break;
      expanded += result.value.byteLength;
      if (expanded > MAX_EXPANDED_BYTES) {
        throw new Error('Snapshot expands beyond the 400 MB limit. Publish a smaller time range.');
      }
      text.push(textDecoder.decode(result.value, { stream: true }));
    }
    text.push(textDecoder.decode());
    return {
      text: text.join(''),
      byteSize: encodedGzip && statedSize > 0 ? statedSize : transferred,
    };
  } finally {
    try { await streamReader?.cancel(); } catch { /* Preserve the download error. */ }
    try { await reader.cancel(); } catch { /* The decompression pipe may own cancellation. */ }
    reader.releaseLock();
  }
}

/** Load a published JSON or gzip snapshot without loading the DuckDB engine. */
export async function loadSnapshot(url, onProgress = () => {}) {
  let address;
  try { address = new URL(url, globalThis.location?.href); } catch {
    throw new Error('The snapshot URL is invalid. Configure an HTTP or HTTPS download URL.');
  }
  if (!['http:', 'https:'].includes(address.protocol)) {
    throw new Error('Snapshots must use an HTTP or HTTPS download URL.');
  }
  onProgress('Downloading the published telemetry snapshot…');
  let response;
  try {
    response = await fetch(address.href, { mode: 'cors', credentials: 'omit' });
  } catch {
    throw new Error('The telemetry snapshot could not be downloaded. Check the URL and connection; its host must allow cross-origin (CORS) downloads from this dashboard and must not require a sign-in.');
  }
  if (!response.ok) {
    throw new Error(`Snapshot download failed (HTTP ${response.status}). Use a public direct-download URL that does not require sign-in.`);
  }
  let downloaded;
  try { downloaded = await readSnapshotText(response, onProgress); } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`The snapshot download could not be read. ${message}`);
  }
  onProgress('Validating the telemetry snapshot…');
  let payload;
  try { payload = JSON.parse(downloaded.text); } catch {
    throw new Error('The snapshot is not valid JSON. Use an exported suncet-beacon-v1 .json or .json.gz file and a direct-download URL, not a sharing preview or sign-in page.');
  }
  const dataset = parseSnapshot(payload);
  let basename = address.pathname.split('/').filter(Boolean).at(-1) || 'Telemetry snapshot';
  try { basename = decodeURIComponent(basename); } catch { /* Keep malformed URL encodings literal. */ }
  return { ...dataset, filename: payload.title?.trim() || basename, byteSize: downloaded.byteSize };
}

/** Quote database identifiers independently from SQL values. */
export function quoteIdentifier(identifier) {
  if (typeof identifier !== 'string' || !identifier || identifier.includes('\0')) {
    throw new Error('Invalid database identifier.');
  }
  return `"${identifier.replaceAll('"', '""')}"`;
}

/**
 * Min/max sampling preserves spikes and chronological order. Missing-data
 * boundaries and state transitions are retained even if that exceeds the
 * display budget: inventing a continuous line or hiding a mode change is worse
 * than rendering a few more points. State values are never averaged.
 */
export function sampleSeries(xs, ys, budget = 1200, kind = 'number') {
  if (xs.length !== ys.length) throw new Error('Series coordinates must have equal lengths.');
  if (!xs.length) return { x: [], y: [] };
  const limit = Number.isFinite(budget) ? Math.max(4, Math.floor(budget)) : 1200;
  const keep = new Set([0, xs.length - 1]);
  const missing = value => value === null || value === undefined ||
    (typeof value === 'number' && !Number.isFinite(value));

  if (kind === 'state' || kind === 'boolean' || kind === 'enum') {
    for (let index = 1; index < ys.length; index += 1) {
      const previous = missing(ys[index - 1]) ? null : ys[index - 1];
      const current = missing(ys[index]) ? null : ys[index];
      if (previous !== current) {
        keep.add(index - 1);
        keep.add(index);
      }
    }
  } else if (xs.length <= limit) {
    return { x: Array.from(xs), y: Array.from(ys, value => numericValue(value)) };
  } else {
    // Preserve both sides of every gap so plotting with connectgaps:false is faithful.
    for (let index = 1; index < ys.length; index += 1) {
      if ((numericValue(ys[index - 1]) === null) !== (numericValue(ys[index]) === null)) {
        keep.add(index - 1);
        keep.add(index);
      }
    }
    const buckets = Math.max(1, Math.floor((limit - 2) / 2));
    const interior = xs.length - 2;
    for (let bucket = 0; bucket < buckets; bucket += 1) {
      const start = 1 + Math.floor(bucket * interior / buckets);
      const end = 1 + Math.floor((bucket + 1) * interior / buckets);
      let minimumIndex = -1;
      let maximumIndex = -1;
      let minimum = Infinity;
      let maximum = -Infinity;
      for (let index = start; index < end; index += 1) {
        const value = numericValue(ys[index]);
        if (value === null) continue;
        if (value < minimum) { minimum = value; minimumIndex = index; }
        if (value > maximum) { maximum = value; maximumIndex = index; }
      }
      if (minimumIndex >= 0) keep.add(minimumIndex);
      if (maximumIndex >= 0) keep.add(maximumIndex);
    }
  }
  const indices = [...keep].sort((left, right) => left - right);
  const isState = ['state', 'boolean', 'enum'].includes(kind);
  return {
    x: indices.map(index => xs[index]),
    y: indices.map(index => isState ? (missing(ys[index]) ? null : ys[index]) : numericValue(ys[index])),
  };
}

function plainRows(table) {
  const names = table.schema.fields.map(field => field.name);
  return table.toArray().map(row => Object.fromEntries(names.map(name => {
    const value = row[name];
    return [name, typeof value === 'bigint' ? Number(value) : value];
  })));
}

function qualifiedName(table) {
  return `${quoteIdentifier(table.table_schema)}.${quoteIdentifier(table.table_name)}`;
}

/**
 * Open a user-selected snapshot entirely in this browser. The filename and
 * telemetry are never uploaded. All DuckDB resources close after materializing
 * the curated rows, allowing another file to be opened without retaining workers.
 */
export async function loadDatabase(file, onProgress = () => {}) {
  if (!file || typeof file.name !== 'string' || typeof file.size !== 'number') {
    throw new Error('Choose a local DuckDB database file.');
  }
  if (file.size === 0) throw new Error('This database file is empty. Choose a completed telemetry snapshot.');

  let database;
  let connection;
  let worker;
  let workerURL;
  let dbVersion = '';
  let phase = 'engine';
  try {
    onProgress('Loading the browser DuckDB engine…');
    const duckdb = await import(`${DUCKDB_PACKAGE}/+esm`);
    const bundle = await duckdb.selectBundle({
      mvp: {
        mainModule: `${DUCKDB_PACKAGE}/dist/duckdb-mvp.wasm`,
        mainWorker: `${DUCKDB_PACKAGE}/dist/duckdb-browser-mvp.worker.js`,
      },
      eh: {
        mainModule: `${DUCKDB_PACKAGE}/dist/duckdb-eh.wasm`,
        mainWorker: `${DUCKDB_PACKAGE}/dist/duckdb-browser-eh.worker.js`,
      },
    });
    workerURL = URL.createObjectURL(new Blob(
      [`importScripts(${JSON.stringify(bundle.mainWorker)});`],
      { type: 'text/javascript' },
    ));
    worker = new Worker(workerURL);
    database = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(duckdb.LogLevel.WARNING), worker);
    await database.instantiate(bundle.mainModule, bundle.pthreadWorker);
    dbVersion = await database.getVersion();
    phase = 'open';
    onProgress('Opening the local database read-only…');
    await database.registerFileHandle('telemetry.duckdb', file, duckdb.DuckDBDataProtocol.BROWSER_FILEREADER, true);
    await database.open({
      path: 'telemetry.duckdb',
      accessMode: duckdb.DuckDBAccessMode.READ_ONLY,
      query: { castBigIntToDouble: true },
    });
    connection = await database.connect();
    phase = 'query';
    onProgress('Finding the beacon telemetry table…');
    const tables = plainRows(await connection.query(`
      SELECT table_schema, table_name FROM information_schema.tables
      WHERE table_type = 'BASE TABLE'
        AND table_schema NOT IN ('information_schema', 'pg_catalog')
      ORDER BY table_schema, table_name
    `));
    const schema = plainRows(await connection.query(`
      SELECT table_schema, table_name, column_name
      FROM information_schema.columns
      WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
      ORDER BY table_schema, table_name, ordinal_position
    `));
    const catalogTable = tables.find(table => table.table_name === '_apid_catalog');
    const catalog = catalogTable
      ? plainRows(await connection.query(`SELECT * FROM ${qualifiedName(catalogTable)} LIMIT 10000`))
      : [];
    const beaconCatalog = catalog.filter(row => Number(row.apid) === 1 || /beacon/i.test(String(row.packet_name ?? '')));
    const candidates = tables.map(table => {
      const columns = schema.filter(column => column.table_schema === table.table_schema && column.table_name === table.table_name)
        .map(column => column.column_name);
      const beaconColumns = columns.filter(column => column.startsWith('beac_')).length;
      const entry = beaconCatalog.find(row => row.table_name === table.table_name);
      return { ...table, columns, score: beaconColumns + (entry ? 10000 : 0) + (entry && Number(entry.apid) === 1 ? 10000 : 0) };
    }).filter(table => table.score > 0).sort((left, right) => right.score - left.score);
    const beacon = candidates[0];
    if (!beacon) throw new Error('No beacon telemetry table was found. Choose a SunCET telemetry DuckDB file containing APID 1 or beac_* columns.');

    const available = new Set(beacon.columns);
    const signalFields = SIGNALS.flatMap(signal => [signal.field, ...(signal.fields ?? []), ...(signal.dependencies ?? [])])
      .filter(field => typeof field === 'string');
    const selected = [...new Set([...CONTEXT_FIELDS, ...signalFields])].filter(field => available.has(field));
    if (!signalFields.some(field => available.has(field))) {
      throw new Error(`Beacon table ${beacon.table_name} has no recognized dashboard telemetry points. Check that this file uses the SunCET telemetry schema.`);
    }
    const countResult = plainRows(await connection.query(`SELECT count(*) AS row_count FROM ${qualifiedName(beacon)}`));
    const rowCount = Number(countResult[0].row_count);
    if (!rowCount) throw new Error(`Beacon table ${beacon.table_name} is empty. Choose a snapshot with decoded beacon packets.`);
    if (rowCount > MAX_ROWS) {
      throw new Error(`This file contains ${rowCount.toLocaleString()} beacon rows. The browser dashboard supports up to ${MAX_ROWS.toLocaleString()} rows; create a smaller snapshot for the time range you want to explore.`);
    }
    onProgress(`Reading ${rowCount.toLocaleString()} beacon packets…`);
    const orderField = ['packet_index', 'source_packet_index'].find(field => available.has(field));
    const order = orderField ? ` ORDER BY ${quoteIdentifier(orderField)}` : '';
    const rows = plainRows(await connection.query(
      `SELECT ${selected.map(quoteIdentifier).join(', ')} FROM ${qualifiedName(beacon)}${order} LIMIT ${MAX_ROWS + 1}`,
    ));
    if (rows.length > MAX_ROWS) throw new Error('This snapshot exceeds the browser row limit. Create a smaller telemetry snapshot.');
    return { rows, columns: beacon.columns, catalog, tableName: beacon.table_name, filename: file.name, dbVersion };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (phase === 'engine') {
      throw new Error(`The browser DuckDB engine could not load. An internet connection is needed to download its pinned runtime. ${message}`);
    }
    if (phase === 'open') {
      throw new Error(`This file could not be opened by browser DuckDB ${dbVersion}. Use a complete, checkpointed DuckDB snapshot. A file created by a newer DuckDB version may need a compatible export. ${message}`);
    }
    throw error;
  } finally {
    try { await connection?.close(); } catch { /* Keep the original result/error. */ }
    try { await database?.terminate(); } catch { /* Worker termination below is the final fallback. */ }
    worker?.terminate();
    if (workerURL) URL.revokeObjectURL(workerURL);
  }
}
