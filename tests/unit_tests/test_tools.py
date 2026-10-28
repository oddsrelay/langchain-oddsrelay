"""What the tools do with the client: the token cap, the key, errors, trimming and paging. No network."""

from __future__ import annotations

import http.client
import json
import re
import urllib.parse
from pathlib import Path

import pytest
from langchain_core.messages import ToolCall

import langchain_oddsrelay
from langchain_oddsrelay import (
    OddsRelayEventOdds,
    OddsRelayEvents,
    OddsRelayOdds,
    OddsRelaySports,
    OddsRelayToolkit,
)

from ..stub import KEY, by_path, fixture

ODDS = {"type": "standard", "sport": "tennis", "bookmakers": ["ladbrokes", "william_hill"]}
EVENT = {"id": "or_evt_da71d5873e85"}


def q(req) -> dict:
    """The query a request sent."""
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(req.full_url).query))


def quoted(cost: int):
    """Replies with a quote of `cost` tokens to quote=true, and from the fixtures otherwise."""
    quote = {"cost": cost, "remaining": 5000, "rate_card": "2026-09"}
    return lambda u: (200, quote, {}) if "quote=true" in u.query else by_path(u)


def call(tool, args) -> tuple[dict, str]:
    """Calls the tool the way an agent does, and returns the reply's JSON and status."""
    msg = tool.invoke(ToolCall(name=tool.name, args=args, id="1", type="tool_call"))
    return json.loads(msg.content), msg.status


@pytest.mark.parametrize(
    "tool, args, sent, hint",
    [
        (OddsRelayOdds, ODDS, {"quote": "true", "sports": "tennis", "bookmakers": "ladbrokes,william_hill"}, "fewer"),
        (OddsRelayEventOdds, {**EVENT, "bookmakers": ["bet600"]}, {"quote": "true"}, "priced whole"),
    ],
)
def test_a_spending_call_is_priced_first_and_refused_above_the_cap(opener, tool, args, sent, hint):
    opener.route = quoted(900)
    body, status = call(tool(api_key=KEY), args)
    assert status == "error"
    assert (body["error"], body["cost"], body["max_tokens"], body["remaining"]) == ("over_max_tokens", 900, 500, 5000)
    assert "priced at 900 tokens" in body["message"] and hint in body["message"]
    [req] = opener.requests  # the free quote only: the call itself was never made
    assert q(req) == sent
    assert req.unredirected_hdrs["Authorization"] == f"Bearer {KEY}"


def test_a_call_priced_at_the_cap_goes_ahead_and_reports_its_tokens(opener):
    opener.route = quoted(500)
    body, status = call(OddsRelayOdds(api_key=KEY), ODDS)
    assert status == "success"
    assert [q(r).get("quote") for r in opener.requests] == ["true", None]
    assert body["tokens"] == {"cost": 12, "remaining": 988}
    assert body["events"] == fixture("v2-odds-standard.json")["data"]


def test_the_model_cannot_raise_the_cap(opener):
    for tool in (OddsRelayOdds(api_key=KEY), OddsRelayEventOdds(api_key=KEY)):
        assert "max_tokens" not in tool.tool_call_schema.model_json_schema()["properties"]
    opener.route = quoted(900)
    body, status = call(OddsRelayOdds(api_key=KEY), {**ODDS, "max_tokens": 10**9})
    assert status == "error" and body["max_tokens"] == 500 and len(opener.requests) == 1


def test_the_cap_cannot_be_switched_off_once_the_tool_is_made():
    tool = OddsRelayOdds(api_key=KEY)
    for off in (None, -1):
        with pytest.raises(ValueError, match="max_tokens"):
            tool.max_tokens = off
    assert tool.max_tokens == 500


def test_the_toolkit_hands_one_key_and_one_cap_to_every_tool():
    tools = OddsRelayToolkit(api_key=KEY, max_tokens=50).get_tools()
    assert [t.name for t in tools] == [
        "oddsrelay_sports",
        "oddsrelay_events",
        "oddsrelay_bookmakers",
        "oddsrelay_odds",
        "oddsrelay_event_odds",
    ]
    assert {t.api_key.get_secret_value() for t in tools} == {KEY}
    assert [t.max_tokens for t in tools if isinstance(t, (OddsRelayOdds, OddsRelayEventOdds))] == [50, 50]


def test_a_400_reaches_the_model_with_its_code_and_the_accepted_keys(opener):
    opener.route = lambda u: (400, fixture("error-400.unknown_bookmaker.json"), {})
    body, status = call(OddsRelayOdds(api_key=KEY), {**ODDS, "bookmakers": ["nope"]})
    assert status == "error"
    assert (body["error"], body["status"]) == ("unknown_bookmaker", 400) and "bet365" in body["valid"]
    assert body["request_id"].startswith("or_req_")
    [req] = opener.requests  # a bad filter fails on the free quote
    assert q(req)["quote"] == "true"


def test_a_402_keeps_the_cost_the_balance_and_the_reset(opener):
    no_tokens = (402, fixture("error-402.insufficient_tokens.json"), {})
    opener.route = lambda u: by_path(u) if "quote=true" in u.query else no_tokens
    body, status = call(OddsRelayEventOdds(api_key=KEY), EVENT)
    assert status == "error"
    assert (body["error"], body["cost"], body["remaining"]) == ("insufficient_tokens", 0, 0)
    assert body["resets_at"] == "2026-10-01T00:00:00Z" and body["request_id"].startswith("or_req_")


def test_the_key_never_reaches_a_reply_an_error_or_a_repr(opener):
    echo = {"error": {"code": "bad_request", "message": f"echo {KEY}", "request_id": KEY, "valid": [KEY]}}
    echoed = {"meta": {}, "data": [{"key": KEY}]}
    for route in (lambda u: (400, echo, {}), lambda u: by_path(u) if "quote=true" in u.query else (200, echoed, {})):
        opener.route = route
        for tool, args in [(OddsRelaySports(api_key=KEY), {}), (OddsRelayOdds(api_key=KEY), ODDS)]:
            msg = tool.invoke(ToolCall(name=tool.name, args=args, id="1", type="tool_call"))
            assert KEY not in msg.content and "[hidden]" in msg.content
            assert KEY not in repr(tool)
    assert KEY not in repr(OddsRelayToolkit(api_key=KEY))


def test_a_missing_or_empty_key_fails_when_the_tool_is_made(monkeypatch):
    monkeypatch.delenv("ODDSRELAY_KEY", raising=False)
    for make in (OddsRelaySports, OddsRelayOdds, OddsRelayToolkit):
        with pytest.raises(ValueError, match="ODDSRELAY_KEY"):
            make()
    with pytest.raises(ValueError, match="ODDSRELAY_KEY"):
        OddsRelayOdds(api_key=" \n")
    monkeypatch.setenv("ODDSRELAY_KEY", f" {KEY}\n")
    assert OddsRelayOdds().api_key.get_secret_value() == KEY  # stripped, as the client strips it


def test_event_odds_narrows_the_reply_to_the_named_bookmakers_and_markets(opener):
    body, status = call(OddsRelayEventOdds(api_key=KEY), {**EVENT, "bookmakers": ["betgoodwin"], "markets": ["h2h"]})
    assert status == "success"
    assert [b["key"] for b in body["bookmakers"]] == ["betgoodwin"]
    assert [m["key"] for m in body["bookmakers"][0]["markets"]] == ["h2h"]
    assert all({"bookmakers", "markets"}.isdisjoint(q(r)) for r in opener.requests)  # the route takes neither
    body, _ = call(OddsRelayEventOdds(api_key=KEY), {**EVENT, "limit": 1})
    assert len(body["bookmakers"]) == 1 and body["more"].startswith("1 more bookmakers not shown")
    assert body["more"].endswith("(a new call, charged again)")


def test_event_odds_leaves_out_venues_with_none_of_the_named_markets(opener):
    event = fixture("v2-odds-event.json")
    bet600, betgoodwin = event["bookmakers"]
    event["bookmakers"] = [{**bet600, "markets": bet600["markets"][:1]}] * 12 + [betgoodwin]  # 12 with h2h only
    opener.route = lambda u: by_path(u) if "quote=true" in u.query else (200, event, {})
    body, status = call(OddsRelayEventOdds(api_key=KEY), {**EVENT, "markets": ["correct_score"]})
    assert status == "success" and "more" not in body
    assert [(b["key"], [m["key"] for m in b["markets"]]) for b in body["bookmakers"]] == [
        ("betgoodwin", ["correct_score"])
    ]


def test_empty_filter_lists_mean_no_filter(opener):
    body, status = call(OddsRelayOdds(api_key=KEY), {**ODDS, "exchanges": [], "markets": []})
    assert status == "success"
    assert all({"exchanges", "markets"}.isdisjoint(q(r)) for r in opener.requests)
    body, status = call(OddsRelayEventOdds(api_key=KEY), {**EVENT, "bookmakers": [], "markets": []})
    assert status == "success" and len(body["bookmakers"]) == 2


def test_a_plain_string_goes_to_the_first_argument(opener):
    assert json.loads(OddsRelayEvents(api_key=KEY).invoke("tennis"))["events"]
    assert q(opener.requests[-1])["sport"] == "tennis"
    assert json.loads(OddsRelaySports(api_key=KEY).invoke("uk"))["sports"]
    assert q(opener.requests[-1])["region"] == "uk"


def test_handle_tool_error_takes_langchains_other_forms(opener):
    opener.route = quoted(900)
    for handler, content in [(lambda e: "refused", "refused"), ("Sorry, the odds tool failed.", None)]:
        msg = OddsRelayOdds(api_key=KEY, handle_tool_error=handler).invoke(
            ToolCall(name="oddsrelay_odds", args=ODDS, id="1", type="tool_call")
        )
        assert msg.status == "error" and msg.content == (content or handler)


def test_paged_tools_reach_every_row_when_limit_cuts_a_page_short(opener):
    pages = {None: ([{"id": f"a{i}"} for i in range(3)], "P2"), "P2": ([{"id": "b0"}], None)}

    def route(u):
        rows, nxt = pages[dict(urllib.parse.parse_qsl(u.query)).get("cursor")]
        return 200, {"meta": {"next_cursor": nxt}, "data": rows}, {}

    opener.route = route
    seen, cursor = [], None
    for _ in range(5):
        args = {"sport": "soccer", "limit": 2, **({"cursor": cursor} if cursor else {})}
        body, _ = call(OddsRelayEvents(api_key=KEY), args)
        seen += [r["id"] for r in body["events"]]
        cursor = body.get("next_cursor")
        if not cursor:
            break
    assert seen == ["a0", "a1", "a2", "b0"]


def fails(err: Exception):
    """A route that fails on the way, as the network or a cut-off reply does."""

    def route(u):
        raise err

    return route


@pytest.mark.parametrize(
    "route",
    [fails(OSError("connection refused")), fails(http.client.IncompleteRead(b"")), lambda u: (200, b"<html>", {})],
    ids=["no-connection", "cut-off", "not-json"],
)
def test_a_failure_on_the_way_is_a_tool_error_and_the_run_goes_on(opener, route):
    opener.route = route
    body, status = call(OddsRelaySports(api_key=KEY), {})
    assert status == "error" and body["error"] == "request_failed"
    assert "charged" not in body["message"]  # a free tool spends nothing


def test_a_spending_call_lost_after_its_quote_says_it_may_have_been_charged(opener):
    lost = fails(TimeoutError("timed out"))
    opener.route = lambda u: by_path(u) if "quote=true" in u.query else lost(u)
    body, status = call(OddsRelayOdds(api_key=KEY), ODDS)
    assert status == "error" and body["error"] == "request_failed"
    assert "TimeoutError: timed out" in body["message"] and "may have been charged" in body["message"]
    assert [q(r).get("quote") for r in opener.requests] == ["true", None]


def test_the_version_matches_pyproject():
    text = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    assert re.search(r'^version = "(.+)"$', text, re.M).group(1) == langchain_oddsrelay.__version__
