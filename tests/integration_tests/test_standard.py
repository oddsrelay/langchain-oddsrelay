"""LangChain's standard integration tests for tools, once per tool, against the stubbed replies (no network).

ODDSRELAY_LIVE=1, with ODDSRELAY_KEY set, runs the same tests against the API instead. By hand only, never in CI,
and pytest needs --force-enable-socket for it. The two spending tools quote first and keep to the default cap.
"""

from __future__ import annotations

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
        return {} if LIVE else {"api_key": KEY}  # live: the key comes from ODDSRELAY_KEY

    @pytest.fixture(autouse=True)
    def _replies(self, request):
        if not LIVE:
            request.getfixturevalue("opener")


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
    tool_invoke_params_example = {"id": "or_evt_da71d5873e85", "markets": ["h2h"], "limit": 3}
