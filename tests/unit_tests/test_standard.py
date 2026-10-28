"""LangChain's standard unit tests for tools, once per tool."""

from __future__ import annotations

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_oddsrelay import (
    OddsRelayBookmakers,
    OddsRelayEventOdds,
    OddsRelayEvents,
    OddsRelayOdds,
    OddsRelaySports,
)

from ..stub import KEY


class _Key:
    tool_constructor_params = {"api_key": KEY}
    init_from_env_params = ({"ODDSRELAY_KEY": KEY}, {}, {"api_key": KEY})


class TestSports(_Key, ToolsUnitTests):
    tool_constructor = OddsRelaySports
    tool_invoke_params_example = {"limit": 5}


class TestEvents(_Key, ToolsUnitTests):
    tool_constructor = OddsRelayEvents
    tool_invoke_params_example = {"sport": "tennis"}


class TestBookmakers(_Key, ToolsUnitTests):
    tool_constructor = OddsRelayBookmakers


class TestOdds(_Key, ToolsUnitTests):
    tool_constructor = OddsRelayOdds
    tool_invoke_params_example = {"type": "standard", "sport": "tennis", "bookmakers": ["william_hill"]}


class TestEventOdds(_Key, ToolsUnitTests):
    tool_constructor = OddsRelayEventOdds
    tool_invoke_params_example = {"id": "or_evt_da71d5873e85", "markets": ["h2h"]}
