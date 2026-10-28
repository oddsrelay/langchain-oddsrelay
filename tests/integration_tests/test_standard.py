"""LangChain's standard integration tests for tools, once per tool, against the stubbed replies (no network).

ODDSRELAY_LIVE=1, with ODDSRELAY_KEY set, runs the same tests against the API instead. By hand only, never in CI,
and pytest needs --force-enable-socket for it. The two spending tools quote first and keep to the default cap.
"""

from __future__ import annotations

import functools
import json
import os

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_oddsrelay import (
    OddsRelayBookmakers,
    OddsRelayEventOdds,
    OddsRelayEvents,
    OddsRelayOdds,
    OddsRelaySports,
)

from ..stub import KEY

LIVE = os.environ.get("ODDSRELAY_LIVE") == "1"


class _Replies:
    @property
    def tool_constructor_params(self) -> dict:
        # A refused call raises instead of coming back as a tool error, so the test fails on it.
        return {"handle_tool_error": False, **({} if LIVE else {"api_key": KEY})}  # live: the key is ODDSRELAY_KEY

    @pytest.fixture(autouse=True)
    def _replies(self, request):
        if not LIVE:
            request.getfixturevalue("opener")


@functools.cache
def _event_on_the_board() -> str:
    """Live: an event on the board now, from oddsrelay_events, which costs no tokens."""
    page = OddsRelayEvents(handle_tool_error=False).invoke({"sport": "tennis", "limit": 1})
    return json.loads(page)["events"][0]["id"]


class TestSports(_Replies, ToolsIntegrationTests):
    tool_constructor = OddsRelaySports
    tool_invoke_params_example = {"limit": 5}


class TestEvents(_Replies, ToolsIntegrationTests):
    tool_constructor = OddsRelayEvents
    tool_invoke_params_example = {"sport": "tennis", "limit": 5}


class TestBookmakers(_Replies, ToolsIntegrationTests):
    tool_constructor = OddsRelayBookmakers


class TestOdds(_Replies, ToolsIntegrationTests):
    tool_constructor = OddsRelayOdds
    tool_invoke_params_example = {"type": "standard", "sport": "tennis", "bookmakers": ["william_hill"], "limit": 3}


class TestEventOdds(_Replies, ToolsIntegrationTests):
    tool_constructor = OddsRelayEventOdds

    @property
    def tool_invoke_params_example(self) -> dict:
        event = _event_on_the_board() if LIVE else "or_evt_da71d5873e85"  # stubbed: the contract's example event
        return {"id": event, "markets": ["h2h"], "limit": 3}
