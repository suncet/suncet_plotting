"""Exercise the launcher's actual HTTP routes using only temporary fixture data."""

import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SPEC = importlib.util.spec_from_file_location("dashboard_serve", Path(__file__).parents[1] / "serve.py")
serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        (self.assets / "index.html").write_text("<h1>Dashboard</h1>")
        (self.assets / "app.mjs").write_text("export const ready = true;")
        (self.assets / "secret.txt").write_text("not an asset")
        (self.assets / "site-config.json").write_text('{"mode":"public","catalogUrl":"https://example.com/catalog.json"}')
        self.database = self.root / "selected.duckdb"
        self.payload = b"test-only-database-bytes"
        self.database.write_bytes(self.payload)
        self.server = serve.make_server(self.database, port=0, asset_directory=self.assets)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.directory.cleanup()

    def request(self, route, **kwargs):
        return urlopen(Request(self.base + route, **kwargs), timeout=5)

    def test_assets_and_private_database(self):
        with self.request("/") as response:
            self.assertIn(b"Dashboard", response.read())
            self.assertIn("text/html", response.headers["Content-Type"])
        with self.request("/app.mjs?v=1") as response:
            self.assertIn("text/javascript", response.headers["Content-Type"])
        with self.request("/local-config.json") as response:
            self.assertEqual(json.load(response), {
                "filename": "selected.duckdb", "url": "local-database.duckdb",
            })
        with self.request("/local-database.duckdb") as response:
            self.assertEqual(response.read(), self.payload)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(int(response.headers["Content-Length"]), len(self.payload))
        self.assertEqual(self.database.read_bytes(), self.payload)

    def test_head_and_byte_ranges(self):
        with self.request("/local-database.duckdb", method="HEAD") as response:
            self.assertEqual(response.read(), b"")
            self.assertEqual(int(response.headers["Content-Length"]), len(self.payload))
        for header, expected in [
            ("bytes=2-5", self.payload[2:6]),
            ("bytes=5-", self.payload[5:]),
            ("bytes=-5", self.payload[-5:]),
        ]:
            with self.subTest(header=header):
                with self.request("/local-database.duckdb", headers={"Range": header}) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.read(), expected)
        for header in ["bytes=999-", "bytes=5-2", "bytes=-0", "bytes=0-1,3-4", "bytes=-"]:
            with self.subTest(header=header):
                with self.assertRaises(HTTPError) as error:
                    self.request("/local-database.duckdb", headers={"Range": header})
                self.assertEqual(error.exception.code, 416)

    def test_launcher_always_returns_local_site_config(self):
        for database in [self.database, None]:
            self.server.database = database
            with self.subTest(database=database):
                with self.request("/site-config.json") as response:
                    self.assertEqual(json.load(response), {"mode": "local", "catalogUrl": ""})
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertIn("application/json", response.headers["Content-Type"])
                with self.request("/site-config.json", method="HEAD") as response:
                    self.assertEqual(response.read(), b"")
                    self.assertGreater(int(response.headers["Content-Length"]), 0)

    def test_restricts_routes_and_host(self):
        for route in [
            "/secret.txt", "/serve.py", "/assets/", "/../selected.duckdb",
            "/%2e%2e/selected.duckdb", "/selected.duckdb", "/local-database.duckdb/extra",
        ]:
            with self.subTest(route=route):
                with self.assertRaises(HTTPError) as error:
                    self.request(route)
                self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError) as error:
            self.request("/local-database.duckdb", headers={"Host": "unrelated.example"})
        self.assertEqual(error.exception.code, 403)

    def test_database_routes_disabled_without_selected_database(self):
        self.server.database = None
        for route in ["/local-config.json", "/local-database.duckdb"]:
            with self.assertRaises(HTTPError) as error:
                self.request(route)
            self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
