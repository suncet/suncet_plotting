#!/usr/bin/env python3
"""Serve the telemetry dashboard and, optionally, one explicitly selected database."""

import argparse
import json
import re
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ASSET_DIRECTORY = Path(__file__).resolve().parent
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.mjs": ("app.mjs", "text/javascript; charset=utf-8"),
    "/data.mjs": ("data.mjs", "text/javascript; charset=utf-8"),
    "/signals.mjs": ("signals.mjs", "text/javascript; charset=utf-8"),
    "/plot-policy.mjs": ("plot-policy.mjs", "text/javascript; charset=utf-8"),
    "/assets/suncet-logo.png": ("assets/suncet-logo.png", "image/png"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class DashboardHandler(BaseHTTPRequestHandler):
    """An explicit route list keeps filesystem paths outside the request API."""

    def do_HEAD(self):
        self._respond(head_only=True)

    def do_GET(self):
        self._respond(head_only=False)

    def _respond(self, head_only):
        # Reject DNS rebinding: only this loopback listener's host is valid.
        port = self.server.server_port
        if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self.send_error(403, "Only localhost requests are accepted")
            return
        route = urlsplit(self.path).path
        if route == "/site-config.json":
            self._send_json({"mode": "local", "catalogUrl": ""}, head_only)
            return
        if route == "/local-config.json" and self.server.database is not None:
            self._send_json({
                "filename": self.server.database.name,
                "url": "local-database.duckdb",
            }, head_only)
            return
        if route == "/local-database.duckdb" and self.server.database is not None:
            self._send_file(self.server.database, "application/octet-stream", head_only, private=True)
            return
        if route in ASSETS:
            name, mime = ASSETS[route]
            self._send_file(self.server.asset_directory / name, mime, head_only)
            return
        self.send_error(404, "Not found")

    def _send_json(self, value, head_only):
        content = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(content)

    def _send_file(self, path, mime, head_only, private=False):
        try:
            source = path.open("rb")
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            self.send_error(404, "Not found")
            return
        with source:
            source.seek(0, 2)
            size = source.tell()
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                # A single byte range supports clients that read databases lazily.
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                valid = match is not None and bool(match[1] or match[2]) and size > 0
                if valid:
                    if match[1]:
                        start = int(match[1])
                        end = min(int(match[2]), size - 1) if match[2] else size - 1
                    else:
                        suffix_size = int(match[2])
                        start = max(size - suffix_size, 0)
                    valid = 0 <= start <= end < size
                if not valid:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store" if private else "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not head_only:
                source.seek(start)
                try:
                    if not range_header:
                        shutil.copyfileobj(source, self.wfile, length=1024 * 1024)
                    else:
                        remaining = length
                        while remaining:
                            chunk = source.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Closing a browser tab can cancel an in-flight download.


def make_server(database=None, port=8051, asset_directory=ASSET_DIRECTORY):
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    server.database = Path(database).expanduser().resolve() if database is not None else None
    server.asset_directory = Path(asset_directory)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", type=Path, help="DuckDB file to open automatically")
    parser.add_argument("--port", type=int, default=8051, help="localhost port (default: 8051)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser automatically")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    database = args.database.expanduser().resolve() if args.database else None
    if database is not None:
        if not database.is_file():
            parser.error(f"database file does not exist: {database}")
        try:
            with database.open("rb"):
                pass
        except OSError as error:
            parser.error(f"database file cannot be read: {error}")
    try:
        server = make_server(database, args.port)
    except OSError as error:
        parser.error(f"could not start server: {error}")
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"SunCET telemetry dashboard: {url}", flush=True)
    if database:
        print(f"Selected database: {database.name}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        opener = threading.Timer(0.25, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
