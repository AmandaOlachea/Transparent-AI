"""Local-only web interface for Transparent-AI."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from explanation_engine import explain

ROOT = Path(__file__).resolve().parent

class Handler(BaseHTTPRequestHandler):
    def reply(self, status, payload, content_type="application/json; charset=utf-8"):
        data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/":
            return self.reply(200, (ROOT / "web" / "index.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
        if route == "/health":
            return self.reply(200, {"status": "ok"})
        return self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/explain":
            return self.reply(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 12000:
                return self.reply(413, {"error": "Invalid request size"})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            return self.reply(200, explain(data.get("text")))
        except (ValueError, TypeError, UnicodeDecodeError) as error:
            return self.reply(400, {"error": str(error)})

if __name__ == "__main__":
    print("Transparent-AI running at http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
