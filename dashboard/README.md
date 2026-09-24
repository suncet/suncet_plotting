# SunCET telemetry dashboard

Explore SunCET beacon telemetry in an interactive browser dashboard. Locally, the app reads a selected DuckDB file using DuckDB-Wasm. The public site gives visitors a catalog of published captures to explore, with no file picker or database upload. Plotly draws the charts; no Grafana or database service is required. The browser needs internet access to load pinned JavaScript libraries from their CDNs.

## Run locally

From the repository root, use Python 3:

```sh
python3 dashboard/serve.py '/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/test_data/2026-09-18_xband_bpsk_flight_like_playback/telemetry/2026-09-18_xband_bpsk_flight_like_playback.duckdb'
```

Or launch without a path and choose a `.duckdb` file in the dashboard:

```sh
python3 dashboard/serve.py
```

The launcher opens `http://127.0.0.1:8051/`. Use `--port 8052` to change the port or `--no-browser` to open the page yourself. Press Ctrl+C in the terminal to stop it. Open the page through the launcher, rather than double-clicking the HTML file, because browser modules and database workers require HTTP.

The server uses the Python standard library and binds to `127.0.0.1`. It serves only the dashboard assets and, when a path was supplied, that one database. It never writes to the source file. `/local-config.json` exposes only the selected filename and its local URL; `/local-database.duckdb` serves that file. `/site-config.json` enables local mode regardless of the public configuration stored in the repository. No directories or arbitrary filesystem paths are exposed. Use a completed database or a consistent DuckDB snapshot; a live database being written by another process may have state in its WAL that is absent from the selected file.

## Explore the beacon

An overview and eight groups organize the recognized beacon fields: temperatures; power; modes and system states; attitude and pointing; timing and counters; heaters; NAND storage; and science. The app finds the beacon table from the APID catalog or its `beac_*` fields and displays the supported channels present in that file.

The overview keeps overlaid comparisons. Dedicated pages give most telemetry points their own plots; the Temperatures page has one plot per available sensor, up to all 12 curated channels. Related comparisons stay together: body-rate axes, wheel speeds, Sun-position coordinates, histogram bins, and matching storage read/write pointers. Modes and other discrete states each have their own row. The Modes page separates subsystem power states from the other system states; power states use green for ON and red for OFF.

Electrical panels pair voltage and current for 11 supplies and calculate watts as **V × I from the same database row**, using the decoded engineering values. Battery calculations show **charge power only**: charge-current telemetry does not measure net battery power or discharge power. X-band amplifier current remains separate because the beacon does not include a matching amplifier voltage. Missing measurements stay missing.

**Packet order** is the default horizontal axis so resets and clock jumps remain visible. You can filter to a spacecraft reset and select **Time since boot (s)** or **CCSDS onboard time (s)**. CCSDS time is the raw coarse seconds plus the fine field in milliseconds, with no UTC epoch inferred. Invalid or missing time values are omitted on time axes. Select one reset when comparing elapsed times, since times from multiple resets can overlap.

Drag horizontally across a plot to synchronize the zoom across panels, or use **Reset zoom** to restore the full range. Click a numeric plot's legend to toggle a channel. Search by display name or telemetry field name. **Hide duplicate packets** is enabled by default when that metadata is available; uncheck it to include duplicates. Modes and other discrete states retain their transitions and decoded labels.

Body rates are displayed in **deg/s**, converted from the stored rad/s values by multiplying by 180/π. This small calculation runs in the browser. **Export view CSV** uses the same displayed deg/s units; the original DuckDB and published compressed snapshot retain rad/s.

Numeric plots use min/max downsampling to retain spikes, endpoints, and missing-data boundaries. State sampling preserves every transition instead of averaging state codes. **Export view CSV** exports the current group's matching channels and available derived powers within the selected reset, duplicate filter, axis, and zoom range; its rows are not downsampled. From the overview, the export includes all matching curated channels.

Zooming resamples the original records within the visible interval, restoring detail. The default **Engineering range** fits all physical measurements inside broad per-sensor display bounds, keeping extreme decoded outliers from flattening useful temperature, voltage, and current variations. Temperature fits stay within −20 to +50 °C, or −50 to +100 °C for solar arrays; electrical and attitude channels have their own bounds. Counters, clocks, and storage pointers use a robust data-based range because they have no fixed physical limits.

**Typical range (1–99%)** excludes gross outliers from the scale calculation before fitting the central values. **Full measured range** includes every finite reading on the vertical axis. Panel footers count readings outside the displayed vertical range. CSV retains those readings, with body rates converted to the displayed deg/s units; changing the scale never deletes readings.

The display bounds are editable in [plot-policy.mjs](plot-policy.mjs). They guide plot scaling and are not operating limits or alarm thresholds.

The browser supports up to **500,000 beacon rows** per file and rejects larger captures with a message to create a smaller snapshot. Memory use also depends on database size and channel count. This is a curated beacon viewer; it does not plot every table or field in the database.

Published snapshots also have limits of 100 MiB compressed and 400 MiB expanded JSON; the exporter checks both before writing files. Split large captures by reset when needed.

## Publish captures for visitors

Keep the original DuckDB files in Dropbox or Google Drive and publish compact beacon snapshots on a separate static data host. GitHub contains the dashboard assets and configuration only. Visitors choose a published capture from the catalog; the site downloads its snapshot and plots it in their browser.

| Location | Contents |
| --- | --- |
| Dropbox or Google Drive | Original DuckDB archive |
| Local export directory outside this repository | Prepared snapshot files and catalog |
| Cloudflare R2 | Catalog and compressed snapshots in a private bucket |
| Cloudflare Worker | Read-only public access to those files at a `workers.dev` address |
| GitHub repository and GitHub Pages | Application assets and the catalog URL |

No domain purchase is required. GitHub Pages supplies the dashboard's address. The included [Cloudflare Worker](cloudflare/worker.mjs) reads snapshots from R2 and serves them on its included `workers.dev` address. Cloudflare describes `workers.dev` as suitable for personal or hobby projects that are not business-critical. See [Workers addresses](https://developers.cloudflare.com/workers/configuration/routing/workers-dev/).

For direct R2 access, the built-in `r2.dev` endpoint is rate-limited and intended for development. A custom domain is an optional alternative for direct production access, not a requirement for the Worker approach. See [R2 public buckets](https://developers.cloudflare.com/r2/buckets/public-buckets/) and [current R2 pricing](https://developers.cloudflare.com/r2/pricing/).

Dropbox's `raw=1` links redirect and its shared-link documentation does not establish the CORS behavior needed here. Drive documents `webContentLink` as a browser download link, which also does not establish cross-origin JavaScript access. For this app, use a data host with explicit CORS configuration instead of relying on those shared links. See the [Dropbox link documentation](https://help.dropbox.com/share/force-download) and [Drive file reference](https://developers.google.com/workspace/drive/api/reference/rest/v3/files).

### Set up the R2 bucket

If your R2 subscription is already activated, start at step 2.

1. [Create a Cloudflare account](https://dash.cloudflare.com/sign-up). In the dashboard, open **Storage & databases → R2 → Overview** and complete the R2 subscription checkout. See [Cloudflare's setup guide](https://developers.cloudflare.com/r2/get-started/).
2. Create a bucket named **`suncet-telemetry-public`**, using **Standard** storage and the automatic location. Keep public access disabled.
3. Prepare the snapshots below and upload `catalog.json` and the generated `.json.gz` files directly into the bucket, without a subfolder. Keep their filenames unchanged. The first prepared capture is `2026-09-18-xband-bpsk.json.gz`.
4. Connect the Worker below. No R2 bucket CORS policy, custom domain, or public development URL is needed for this approach.

Uploads through the Cloudflare website require no API credentials in the repository. The Worker accesses the private bucket through a binding; the exported files it serves are publicly downloadable.

### Connect the Worker without buying a domain

If the bucket and files are already uploaded, start here. The [Worker source](cloudflare/worker.mjs) is a complete JavaScript module that can be pasted into Cloudflare's editor.

1. In Cloudflare, open **Workers & Pages → Create application → Start with Hello World! → Get started**. Name the Worker **`suncet-telemetry`** and select **Deploy**. If prompted for a `workers.dev` subdomain, choose an available name; this does not require buying a domain. See [Cloudflare's create-Worker instructions](https://developers.cloudflare.com/kv/get-started/#1-create-a-worker-project).
2. Open the Worker's **Bindings** tab. Select **Add binding → R2 bucket → Add binding**, set **Variable name** to **`TELEMETRY_BUCKET`**, and choose **`suncet-telemetry-public`** from the bucket list. Select **Add binding** to save it. The variable name must match exactly. See [bindings in the dashboard](https://developers.cloudflare.com/kv/get-started/#3-bind-your-worker-to-your-kv-namespace) and [R2 bucket bindings](https://developers.cloudflare.com/r2/api/workers/workers-api-usage/).
3. Select **Edit code**, replace the starter code with the entire contents of [cloudflare/worker.mjs](cloudflare/worker.mjs), and select **Deploy**.
4. Copy the Worker's public address, which looks like `https://suncet-telemetry.YOUR-SUBDOMAIN.workers.dev`. Append **`/catalog.json`** and open that URL. It should show JSON containing `suncet-catalog-v1` and the uploaded capture.
5. Set that full catalog URL in `dashboard/site-config.json`. The current configuration uses `https://suncet-telemetry.jmason86.workers.dev/catalog.json`.

The Worker accepts reads of `catalog.json` and exported snapshot files. It adds CORS headers for `https://suncet.github.io`, `http://127.0.0.1:8051`, and `http://localhost:8051`. If the dashboard moves to a different origin, update the allowed origins in the Worker and redeploy it. The origin must not include `/suncet_plotting/` or another path. R2 bucket CORS settings do not apply to responses served by this Worker.

For deployment from a terminal instead, the included [Wrangler configuration](cloudflare/wrangler.jsonc) sets the same Worker name and bucket binding. From the repository root, run:

```sh
npx wrangler login
npx wrangler deploy --config dashboard/cloudflare/wrangler.jsonc
```

The login opens Cloudflare in a browser. Use the account that owns the bucket. If you chose a different bucket name, change `bucket_name` in the configuration first. Wrangler prints the deployed `workers.dev` address. See [R2 Worker deployment](https://developers.cloudflare.com/r2/get-started/workers-api/).

### Prepare a snapshot

The exporter needs the Python `duckdb` package. Run it locally with a completed source database and an output directory outside the Git repository:

```sh
python3 dashboard/export_snapshot.py '/path/to/telemetry.duckdb' \
  --output-dir '/path/outside/repo/suncet-public-data' \
  --title '2026-09-18 · X-band BPSK flight-like playback' \
  --id 2026-09-18-xband-bpsk
```

The export includes curated telemetry columns, raw engineering values, and decoded state strings. It excludes source paths, packet hashes, and unrelated bookkeeping. The source database remains unchanged. Published snapshots still contain telemetry and are downloadable by visitors, so select the captures and titles intended for publication.

The exporter writes `<id>.json.gz` and `catalog.json`. By default, the ID is a filename-safe form of the title; set `--id 2026-09-18-xband-bpsk` for a stable identifier. Use `yyyy-mm-dd` dates in capture titles. Reusing an ID replaces its snapshot and catalog entry; a new ID appends a capture while preserving the existing entries. Use `--reset 123` to export only a particular spacecraft reset. Exports are limited to 500,000 beacon rows after filtering. This command only prepares local files; it does not upload them.

### Optional: use a different data host or direct R2 access

Skip this section when using the Worker above. These steps apply only to a separate static data host or direct R2 access.

1. Upload the generated catalog and compressed snapshot files to the same directory or prefix on your data host. For direct R2 testing, enable the bucket's public development URL; for direct production access, connect a custom domain you own.
2. Serve the catalog as `application/json` and the `.json.gz` snapshots as `application/gzip`. Leave `Content-Encoding` unset; the app handles the compressed file.
3. Allow CORS requests from the dashboard's origin. For the repository's GitHub Pages site, paste [r2-cors.example.json](r2-cors.example.json) into **R2 bucket → Settings → CORS Policy → JSON**. It allows `GET` and `HEAD` from `https://suncet.github.io` and local previews at `http://127.0.0.1:8051` or `http://localhost:8051`. Replace the Pages origin if you use another domain; the origin must not include `/suncet_plotting/` or any other path. See [R2 CORS configuration](https://developers.cloudflare.com/r2/buckets/cors/).
4. Keep the public HTTPS URL of the catalog for the Pages configuration below. The catalog and snapshots must be accessible without signing in.

The catalog has format `suncet-catalog-v1`, with a `datasets` array containing each capture's `id`, `title`, `url`, `rowCount`, and `bytes`. A relative snapshot URL is resolved against the catalog URL, so keep the generated files together when uploading.

### Deploy the GitHub Pages interface

1. Push the dashboard code to the GitHub repository.
2. In **Settings → Pages**, choose **GitHub Actions** as the source.
3. In **Actions → Publish telemetry dashboard**, select **Run workflow** and then open its deployment URL.

The workflow publishes only the dashboard's HTML, JavaScript, CSS, SunCET logo asset, and generated `site-config.json`. It does not upload databases, snapshots, exports, or the Python launcher. It uses the public catalog URL in the checked-in `dashboard/site-config.json`. To override that URL without editing the file, add an optional `TELEMETRY_CATALOG_URL` repository variable under **Settings → Secrets and variables → Actions → Variables**. The workflow validates the selected URL as HTTPS before publishing.

Publishing is manual (`workflow_dispatch`); committing changes does not deploy them. Once the host is configured, adding captures means uploading the new snapshot files and updated catalog to that host. The dashboard code only needs redeployment when it changes or when you change the catalog URL. See GitHub's [custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Developer checks

Use Python 3 with `duckdb` installed and a current Node.js installation on your PATH. Node.js is only needed for the JavaScript tests, not to run the dashboard.

```sh
python3 -m unittest discover -s dashboard/tests -p 'test_*.py'
node --test tests/*.test.mjs
```

The Python tests use temporary fixtures to check asset routing, local/public configuration separation, response sizes, byte ranges, route restrictions, snapshot exports, and catalog updates. The Node tests check missing values, state labels, identifier quoting, plot grouping and scale policies, downsampling fidelity, and Worker routing, CORS, caching, and compressed snapshot loading. Worker tests use a simulated R2 binding; verify the deployed catalog URL after setup. These tests do not open or modify mission telemetry.
