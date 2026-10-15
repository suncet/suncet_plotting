# SunCET telemetry dashboard

Explore SunCET beacon telemetry in an interactive browser dashboard. Locally, the app reads a selected DuckDB file using DuckDB-Wasm. The public site gives visitors a catalog of published captures and an APID packet selector, with no file picker or database upload. Plotly draws the charts; no Grafana or database service is required. The browser needs internet access to load pinned JavaScript libraries from their CDNs.

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

Select a capture, then choose **APID number · packet name — description**. The selector lists the APIDs included in that capture; old exports remain beacon-only until re-prepared. Only the selected packet snapshot is downloaded. The left navigation changes to categories present in that APID, and non-beacon views show up to 12 separate plots per page with Previous/Next controls. Search covers all telemetry points in the selected APID. APID selection is included in shareable URLs as `apid=27`, for example.

An overview and eight groups organize the recognized beacon fields: temperatures; power; modes and system states; attitude and pointing; timing and counters; heaters; NAND storage; and science. The app finds the beacon table from the APID catalog or its `beac_*` fields and displays the supported channels present in that file.

The overview keeps overlaid comparisons. Dedicated pages give most telemetry points their own plots; the Temperatures page has one plot per available sensor, up to all 12 curated channels. Related comparisons stay together: body-rate axes, wheel speeds, Sun-position coordinates, histogram bins, and matching storage read/write pointers. Modes and other discrete states each have their own row. The Modes page separates subsystem power states from the other system states; power states use green for ON and red for OFF.

Electrical panels pair voltage and current for 11 supplies and calculate watts as **V × I from the same database row**, using the decoded engineering values. Battery calculations show **charge power only**: charge-current telemetry does not measure net battery power or discharge power. X-band amplifier current remains separate because the beacon does not include a matching amplifier voltage. Missing measurements stay missing.

In the paired battery plots, current and power lines use each battery's reported charging-state flag: green for charging, red for discharging, and gray for an unknown state. The legend controls both current and power traces together; the voltage line keeps its usual color. State coloring does not change measurement values or signs. Heater enable timelines use green for NO and red for YES.

**UTC** is the default horizontal axis; **Packet order** remains available. **Time alive (s)** uses the beacon’s cumulative lifetime counter across reboots; **Time since boot (s)** uses its resettable boot counter. **CCSDS onboard time (s)** shows raw coarse seconds plus integer milliseconds. **UTC** follows the realtime display’s configured FSW convention: midnight 2000-01-01 plus the five post-J2000 leap seconds through 2017 (update the shared convention if this changes). UTC is onboard time, not ground reception time. UTC omits clock values outside 2020–2099 so unset clocks do not flatten mission-era plots; the note counts omitted rows. Missing or invalid times are also omitted. Reboots and clock discontinuities break plotted lines and state intervals. UTC zoom is synchronized across panels, and UTC CSV coordinates are ISO 8601 strings ending in `Z`. A supplied `analysis_time_reconstructed` flag is carried into snapshots and identified in the time-axis note.

Drag horizontally across a plot to synchronize the zoom across panels, or use **Reset zoom** to restore the full range. Click a numeric plot's legend to toggle a channel. Search by display name or telemetry field name. **Hide duplicate packets** is enabled by default when that metadata is available; uncheck it to include duplicates. Modes and other discrete states retain their transitions and decoded labels.

Body rates are displayed in **deg/s**, converted from the stored rad/s values by multiplying by 180/π. This small calculation runs in the browser. **Export view CSV** uses the same displayed deg/s units; the original DuckDB and published compressed snapshot retain rad/s.

Numeric plots use min/max downsampling to retain spikes, endpoints, and missing-data boundaries. State sampling preserves every transition instead of averaging state codes. **Export view CSV** exports the current group's matching channels and available derived powers within the selected reset, duplicate filter, axis, and zoom range; its rows are not downsampled. From the overview, the export includes all matching curated channels.

Zooming resamples the original records within the visible interval, restoring detail. The default **Engineering range** fits all physical measurements inside broad per-sensor display bounds, keeping extreme decoded outliers from flattening useful temperature, voltage, and current variations. Temperature fits stay within −20 to +50 °C, −40 to +40 °C for the CSIE detector, or −50 to +100 °C for solar arrays; electrical and attitude channels have their own bounds. Counters and clocks use a robust data-based range. NAND pointers use the full configured partition capacity in Engineering range.

**Typical range (1–99%)** excludes gross outliers from the scale calculation before fitting the central values. **Full measured range** includes every finite reading on the vertical axis. Panel footers count readings outside the displayed vertical range. CSV retains those readings, with body rates converted to the displayed deg/s units; changing the scale never deletes readings.

The display bounds are editable in [plot-policy.mjs](plot-policy.mjs). They guide plot scaling and are not operating limits or alarm thresholds.

NAND read and write pointer plots show native pages on the left and equivalent bytes on the right, using **6,912 bytes per page**. The byte scale follows the page axis through zoom and scale changes; hover labels show both units. Stored values and CSV pointer values remain in pages. Engineering range is fixed at 0–388,352 pages for housekeeping, 0–72,320 for logs, 0–1,087,360 for ADCS, 0–144,640 for Dual-SPS, and 0–2,485,504 for science (also used for CSIE science-image and legacy metadata pointers). Typical range and Full measured range remain available for examining smaller changes or outliers.

The browser supports up to **500,000 beacon rows** per file and rejects larger captures with a message to create a smaller snapshot. Memory use also depends on database size and channel count. The limit applies to each APID snapshot. Public APID fields are explicitly listed in `public-packets.json`, with names and enums derived from the bus 2.0.5 telemetry definitions. All `adcs_*` APIDs, commands, fault protection, and sequence engines remain excluded. ADCS telemetry points within the beacon remain available. The dashboard also hides ADCS APIDs from older catalogs. Packet headers, provenance, checksums, spare fields, and log-event parameter bytes are not plotted. Log events include header fields only. Some unsupported units are left blank rather than inferred.

DSPS definitions use the 6.09 generated decoder: **APID 35 · dsps_data** (science and health), **25 · dsps_param** (parameters), **29 · dsps_time** (timing), and **23 · dsps_log** (sequence counts only; log payload bytes remain excluded). These appear automatically after the corresponding packets have been processed into the capture's database and `prepare_tvac2.py` has regenerated the snapshots/catalog. A capture without DSPS rows has no DSPS selector entries. No website deployment is needed to add those entries; upload the generated snapshots, then the catalog last.

APID 35 currently uses the pipeline's provisional `DSPS_PASS` prefix decoder. Its checksum and unknown tail are excluded. Several temperature and analog-monitor fields have engineering-sounding names but no supplied DSPS conversion functions; these are labeled **decoded units**, preserved unchanged, and do not receive degree-Celsius display limits. Explicit millivolt and milliamp fields retain those units. Header aliases (`dsps_pass`, `dsps_parameter_pkt`) preserve the packet's own time without borrowing a beacon timestamp. Invalid or unset times remain accessible on Packet order.

Published snapshots also have limits of 100 MiB compressed and 512 MiB expanded JSON; the exporter checks both before writing files. The expanded limit accommodates the complete TVAC2 playback. Split larger captures by reset when needed.

## Refresh TVAC2 upload files

After adding data and completing the usual processing-pipeline run, run this from any Terminal directory on Mason’s Mac:

```sh
conda run --no-capture-output -n suncet python "/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/code/suncet_plotting/dashboard/prepare_tvac2.py"
```

This uses the existing `suncet` environment (Python with `duckdb`). It combines the completed TVAC2 beacon batches in the shared database, including incremental `__xband_…` ingestions, preserves valid UTC, and backfills only initial unset clocks with a later anchor in the same boot. Boots without an anchor remain available on the other time axes. Unexpected time regressions stop preparation for inspection; powered-off calendar gaps are never compressed.

The command fetches the live catalog and replaces just the TVAC2 entry, including links to its available APIDs. It scopes every APID to the same capture folder in `_ingestions`, never the whole shared database. Additional APIDs retain their original CCSDS times; beacon corrections are not applied to other packets. Their unset clocks are omitted from UTC plots and remain available by packet order. Reset and time-alive controls are disabled when a packet does not report those values. It stages and validates the export before replacing the corrected database, audit report, and all packet snapshots plus the catalog in `2026-10-07_tvac2/public_dashboard`. Previous versions are saved under `previous_exports`. It checks all successful processing manifests for new or changed raw inputs; the newest incremental run need not list earlier downlinks. Packet indices that restart in each batch receive unique display indices, while the corrected database retains `original_packet_index`. Every selected batch is checked against its ingestion row count. It stops if an input has no matching successful run. The source database is opened read-only.

Upload **all `.json.gz` files** from `public_dashboard` to the R2 bucket root, then upload **`catalog.json` last**, overwriting matching objects. There is now one snapshot per available APID, plus the catalog. Additional APID filenames contain a content hash; old R2 snapshots can remain until you choose to clean them up. The script backs up and removes obsolete generated APID files from the local upload folder. The command does not upload or rerun decoding. No website deployment is needed for a data update.

Use `--database PATH` if the shared database moves, or `--folder PATH` if the TVAC2 folder moves. `--catalog-file PATH` explicitly supplies a catalog for offline preparation; otherwise a catalog download failure stops the command rather than risking a stale catalog. If TVAC2 exceeds the dashboard’s 500,000-row or download-size limits, preparation stops and the capture will need splitting.

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

### CSIE engineering packets

The public exporter also includes CSIE 1.1.8 engineering telemetry, with `CSIE` in each selector title to distinguish packets that share names with the bus. This includes APID **533 · storage_hk**, **525 · flash_hk**, **543 · nand_stream_hk_pkt**, hardware/software/library/memory health, task health and diagnostics, time synchronization, telemetry/table/log health, version, CFI, heater PID, instrument housekeeping, and image metadata. Command (521/524), sequence (519), and fault-protection (540/541) APIDs and related telemetry fields remain excluded. Image payload (536), memory dumps (518), spare fields, checksums, and log parameter bytes are not exported as trend plots.

Only APIDs with ingested rows in the selected capture appear. `prepare_tvac2.py` includes these automatically; upload the snapshots before updating the catalog. Both the CSIE `SHCOARSE`/`SHFINE` secondary header and older named secondary-header columns are supported, without substituting bus/beacon times. Temperatures without a supplied engineering conversion retain decoded units. ADC thermistors use the pipeline's renamed fields rather than their obsolete spare names.

TVAC2 preparation merges beacon playback batches by the spacecraft's cumulative time-alive counter, so a later downlink can fill an earlier gap. Each input batch must still have a nondecreasing counter, and the merged corrected UTC must remain nondecreasing. Original packet indices, overlapping records, and all set onboard clocks are preserved; packet-order indices in the prepared beacon copy describe this merged timeline. Other APIDs retain ingestion order, with UTC plots sorted by their own packet timestamps.
