# suncet_plotting
Routines for making a variety of plots and movies.

The [telemetry dashboard](dashboard/README.md) opens SunCET DuckDB telemetry in an interactive browser view:

```sh
python3 dashboard/serve.py /path/to/telemetry.duckdb
```

For local analysis, you can also launch without a path and choose a file in the page. For public access, visitors choose published captures from a catalog. GitHub Pages hosts the interface; a Cloudflare Worker serves the catalog and compressed snapshots from a private R2 bucket. No domain purchase is needed. Original DuckDB files can remain in Dropbox or Google Drive, and GitHub contains application assets and configuration only. See the [R2, Worker, and publishing instructions](dashboard/README.md#publish-captures-for-visitors).

Explore eight beacon groups, including individual temperature plots, paired voltage/current measurements with calculated power, and mode timelines. The overview retains overlaid comparisons. Default engineering scales keep extreme outliers from flattening the useful measurements while preserving full-range viewing. Body rates display and export in deg/s; original data stays in rad/s. The dashboard also includes synchronized zoom, reset and telemetry point filters, alternative time axes, and CSV export of the selected view.
