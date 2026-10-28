"""The opener fixture: every OddsRelay client the tools make answers from tests/fixtures, so no test reaches the API."""

from __future__ import annotations

import oddsrelay
import pytest

import langchain_oddsrelay

from .stub import Opener


@pytest.fixture
def opener(monkeypatch):
    """The real client, with this stand-in opener and no waiting between retries."""
    op = Opener()
    monkeypatch.setattr(
        langchain_oddsrelay, "OddsRelay", lambda key: oddsrelay.OddsRelay(key, opener=op, sleep=lambda s: None)
    )
    return op
