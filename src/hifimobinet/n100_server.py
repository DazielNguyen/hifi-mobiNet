"""Small LAN-only browser UI using Python's HTTP server and the N100 runtime."""
from __future__ import annotations
import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from hifimobinet.n100_demo import Engine, ROOT, bundle_path, runs_path

ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status, payload, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                             "style-src 'self'; media-src blob:; connect-src 'self'; "
                             "object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, status, data):
            self.send_bytes(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
                            "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ASSETS:
                name, mime = ASSETS[self.path]
                self.send_bytes(200, (ROOT / "demo/n100_web" / name).read_bytes(), mime)
            elif self.path == "/api/status":
                self.send_json(200, {"server": engine.preflight,
                                    "models": [{"id": m["id"], "label": m["label"],
                                                "sample_rate": m["sample_rate"]}
                                               for m in engine.models.values()]})
            elif self.path == "/favicon.ico":
                self.send_bytes(204, b"", "image/x-icon")
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/synthesize":
                self.send_json(404, {"error": "Not found"})
                return
            # Same-origin JSON fetch only. Do not enable cross-origin requests.
            origin = urlsplit(self.headers.get("Origin", ""))
            if origin.scheme != "http" or origin.netloc != self.headers.get("Host"):
                self.send_json(403, {"error": "Open the demo page on this N100 server to generate audio."})
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self.send_json(415, {"error": "Expected JSON"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Invalid request size")
                self.connection.settimeout(15)
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or set(data) != {"text", "model_id", "precision"}:
                    raise ValueError("Expected text, model_id and precision")
                if any(not isinstance(value, str) for value in data.values()):
                    raise ValueError("All request fields must be strings")
            except (ValueError, OSError) as error:
                self.send_json(400, {"error": str(error)})
                return
            try:
                wav, record = engine.synthesize(**data)
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
                return
            except Exception as error:
                self.send_json(503, {"error": str(error)})
                return
            self.send_json(200, {"wav_base64": base64.b64encode(wav).decode("ascii"), "record": record})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8501, type=int)
    args = parser.parse_args()
    print("Checking N100 environment, IR hashes and Harvard-720 frontend...", flush=True)
    engine = Engine(bundle_path(), runs_path())
    with ThreadingHTTPServer((args.host, args.port), handler_for(engine)) as server:
        print(f"N100 DEMO READY on port {args.port}. Logs: {engine.session}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Demo server stopped.")


if __name__ == "__main__":
    main()
