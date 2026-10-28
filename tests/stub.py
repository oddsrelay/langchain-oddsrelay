"""Replies for the tests: a stand-in for urllib's opener that answers from tests/fixtures (the API's own examples)."""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
import urllib.parse
from email.message import Message
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
KEY = "test-key-123"  # not a real key: tests never call the API

ROUTES = {  # path prefix -> the fixture a 200 carries; the first match wins
    "/v2/sports": "v2-sports.page.json",
    "/v2/events": "v2-events.page.json",
    "/v2/bookmakers": "v2-bookmakers.json",
    "/v2/odds/event/": "v2-odds-event.json",
    "/v2/odds/": "v2-odds-standard.json",
}


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def by_path(url: urllib.parse.SplitResult) -> tuple[int, dict, dict]:
    """The default replies: the quote for quote=true, else the path's fixture with its token headers."""
    if "quote=true" in url.query:
        return 200, fixture("quote.json"), {}
    for prefix, name in ROUTES.items():
        if url.path.startswith(prefix):
            return 200, fixture(name), {"X-Tokens-Cost": "12", "X-Tokens-Remaining": "988"}
    return 404, {"error": {"code": "not_found", "message": "no stub for this path"}}, {}


class FakeResponse(io.BytesIO):
    def __init__(self, status: int, headers: Message, body: bytes) -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers


class Opener:
    """Stands in for urllib's opener: answers each request from `route(url)`, gzipped, and records it."""

    def __init__(self, route=by_path) -> None:
        self.route = route
        self.requests = []

    def open(self, req, timeout=None):
        self.requests.append(req)
        status, body, extra = self.route(urllib.parse.urlsplit(req.full_url))
        headers = Message()
        headers["Content-Type"] = "application/json"
        headers["Content-Encoding"] = "gzip"
        for name, value in extra.items():
            headers[name] = value
        raw = gzip.compress(body if isinstance(body, bytes) else json.dumps(body).encode())  # bytes: sent as they are
        if status >= 300:  # urllib raises for these, as the real opener does
            raise urllib.error.HTTPError(req.full_url, status, "", headers, io.BytesIO(raw))
        return FakeResponse(status, headers, raw)
