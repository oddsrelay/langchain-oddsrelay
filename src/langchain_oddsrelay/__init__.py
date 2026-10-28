"""LangChain tools for the OddsRelay API, built on the oddsrelay Python client.

Five tools that only read, and a toolkit that makes all five with one key and one cap. The two tools that spend
tokens, oddsrelay_odds and oddsrelay_event_odds, ask the API for a price first and refuse a call priced above their
max_tokens: a cap the developer sets where the tool is made, which the model can neither see nor raise.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Literal

from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from langchain_core.utils import secret_from_env
from oddsrelay import APIError, OddsRelay, OddsRelayError, QuoteTooHigh, Result
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SecretStr, StringConstraints

__version__ = "0.1.0"
__all__ = [
    "OddsRelayBookmakers",
    "OddsRelayEventOdds",
    "OddsRelayEvents",
    "OddsRelayOdds",
    "OddsRelaySports",
    "OddsRelayToolkit",
    "__version__",
]

DEFAULT_MAX_TOKENS = 500  # the cap when the developer sets no max_tokens: a call quoted above it is not made
REPLY_BUDGET = 30_000  # characters of rows per reply, so one reply fits in an agent's context
NO_KEY = "Set ODDSRELAY_KEY to your OddsRelay API key, or pass api_key."

# Every key the API hands out (sports, venues, markets, events) fits this, and a comma never does, so one value
# stays one value.
Key = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_.-]+$")]


def _checked(key: SecretStr) -> SecretStr:
    """Spaces and newlines around the key are dropped, as the client drops them. An empty key is refused."""
    value = key.get_secret_value().strip()
    if not value:
        raise ValueError(NO_KEY)
    return SecretStr(value)


ApiKey = Annotated[SecretStr, AfterValidator(_checked)]


def _key_field() -> Any:
    # A missing ODDSRELAY_KEY fails here, when the tool is made, not at its first call.
    return Field(default_factory=secret_from_env("ODDSRELAY_KEY", error_message=NO_KEY), validate_default=True)


def _limit(rows: str, default: int) -> Any:
    return Field(default=default, ge=1, le=100, description=f"Most {rows} to return; a note counts the rest")


_REGION = "Region code. Left out, the API uses uk."
_CURSOR = "next_cursor from the previous reply, with the same other arguments"
_FROM = "Earliest start time, ISO 8601"
_TO = "Latest start time, ISO 8601"


class _OddsRelayTool(BaseTool):
    """What every tool shares: the key, errors handed back to the model, and the reply text."""

    model_config = ConfigDict(validate_assignment=True)  # a key or cap changed after the tool is made is checked too

    api_key: ApiKey = _key_field()
    """Your OddsRelay API key. Left out, it is read from ODDSRELAY_KEY."""
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True
    """A refused call (over the cap, a bad filter, a network failure) goes back to the model as a tool error.
    LangChain's other forms work too: False raises the ToolException, and a string or a function words it your way."""
    narrow: ClassVar[str] = ""  # what to change when a call is priced above the cap

    def _run(self, *args: Any, **kwargs: Any) -> str:
        schema = self.get_input_schema()
        if args:  # tool.invoke("tennis"): LangChain passes a plain string alone, meant for the first argument
            kwargs[next(iter(schema.model_fields))] = args[0]
        a = schema.model_validate(kwargs)  # the arguments as attributes, defaults included
        key = self.api_key.get_secret_value()
        try:
            # A fresh client per call: nothing is shared between threads, and no ETag is sent.
            # ponytail: a repeated call is charged again; keep one client per tool if repeats turn out to matter.
            body = self._read(OddsRelay(key), a)
        except Exception as err:  # every failure goes back to the model as a tool error, so the agent's run goes on
            raise ToolException(_text(_explain(err, self), key)) from None
        return _text(body, key)

    def _read(self, client: OddsRelay, args: Any) -> dict:
        raise NotImplementedError


class _SpendingTool(_OddsRelayTool):
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=0)
    """The cap, in tokens: a call quoted above it is not made. Set by you, never by the model."""


class _SportsInput(BaseModel):
    region: Key | None = Field(default=None, description=_REGION)
    cursor: str | None = Field(default=None, description=_CURSOR)
    limit: int = _limit("sports", 50)


class OddsRelaySports(_OddsRelayTool):
    """Sport and competition keys, a page at a time. Costs no tokens."""

    name: str = "oddsrelay_sports"
    description: str = (
        "Lists sport and competition keys (such as soccer or soccer_epl) for the sport argument of oddsrelay_odds "
        "and oddsrelay_events. Costs no tokens. Paged: pass next_cursor back as cursor."
    )
    args_schema: type[BaseModel] = _SportsInput

    def _read(self, client: OddsRelay, a: _SportsInput) -> dict:
        skip, api = _unpack(a.cursor)
        return _page(client.sports(region=a.region, cursor=api).data, skip, api, a.limit, "sports")


class _EventsInput(BaseModel):
    sport: Key = Field(description="Which sport: tennis, or a competition key from oddsrelay_sports")
    region: Key | None = Field(default=None, description=_REGION)
    commence_from: str | None = Field(default=None, description=_FROM)
    commence_to: str | None = Field(default=None, description=_TO)
    cursor: str | None = Field(default=None, description=_CURSOR)
    limit: int = _limit("events", 25)


class OddsRelayEvents(_OddsRelayTool):
    """The events of one sport, with their ids, a page at a time. Costs no tokens."""

    name: str = "oddsrelay_events"
    description: str = (
        "Lists the events of one sport with their ids, for oddsrelay_event_odds. Costs no tokens. "
        "Paged: pass next_cursor back as cursor."
    )
    args_schema: type[BaseModel] = _EventsInput

    def _read(self, client: OddsRelay, a: _EventsInput) -> dict:
        skip, api = _unpack(a.cursor)
        res = client.events(
            a.sport, region=a.region, commence_time_from=a.commence_from, commence_time_to=a.commence_to, cursor=api
        )
        return _page(res.data, skip, api, a.limit, "events")


class _NoInput(BaseModel):
    pass


class OddsRelayBookmakers(_OddsRelayTool):
    """The bookmaker and exchange keys the odds tools take. Costs no tokens."""

    name: str = "oddsrelay_bookmakers"
    description: str = (
        "Lists the bookmaker and exchange keys that oddsrelay_odds and oddsrelay_event_odds take. Costs no tokens."
    )
    args_schema: type[BaseModel] = _NoInput

    def _read(self, client: OddsRelay, a: _NoInput) -> dict:
        return {"venues": client.bookmakers().data["data"]}


class _OddsInput(BaseModel):
    type: Literal["standard", "2up", "dutching", "each-way", "extra-place", "bog"] = Field(
        description="Which matched board: standard, unless the user asks for another"
    )
    sport: Key = Field(description="Which sport: soccer, or a competition key from oddsrelay_sports such as soccer_epl")
    bookmakers: list[Key] = Field(min_length=1, max_length=40, description="Bookmaker keys from oddsrelay_bookmakers")
    exchanges: list[Key] | None = Field(
        default=None, max_length=4, description="Exchange keys; left out, every exchange on the board"
    )
    markets: list[Key] | None = Field(
        default=None, max_length=20, description="Market keys such as h2h; left out, every market on the board"
    )
    commence_from: str | None = Field(default=None, description=_FROM)
    commence_to: str | None = Field(default=None, description=_TO)
    region: Key | None = Field(default=None, description=_REGION)
    limit: int = _limit("events", 10)


class OddsRelayOdds(_SpendingTool):
    """A matched board for one sport at the bookmakers named. Priced first, and refused above max_tokens."""

    name: str = "oddsrelay_odds"
    description: str = (
        "Reads a matched board for one sport at the bookmakers you name: bookmaker back prices beside exchange lay "
        "prices, or on dutching, sets of back prices covering every outcome of an event. Spends tokens: priced "
        "first, and refused when the price is above this tool's cap."
    )
    args_schema: type[BaseModel] = _OddsInput
    narrow: ClassVar[str] = (
        "On a standard or dutching board, naming fewer bookmakers, exchanges or markets can lower the price. The "
        "other boards cost the same whatever the filters, and the kick-off window and limit never change a price."
    )

    def _read(self, client: OddsRelay, a: _OddsInput) -> dict:
        res = client.odds(
            a.type,
            max_tokens=self.max_tokens,
            region=a.region,
            sports=a.sport,
            bookmakers=a.bookmakers,
            exchanges=a.exchanges or None,  # an empty list means no filter, as left out does
            markets=a.markets or None,
            commence_time_from=a.commence_from,
            commence_time_to=a.commence_to,
        )
        return {"meta": res.data["meta"], **_trim(res.data["data"], a.limit, "events"), "tokens": _spent(res)}


class _EventOddsInput(BaseModel):
    id: Key = Field(description="An event id from oddsrelay_events or oddsrelay_odds, such as or_evt_da71d5873e85")
    bookmakers: list[Key] | None = Field(
        default=None, max_length=40, description="Only these bookmaker or exchange keys"
    )
    markets: list[Key] | None = Field(default=None, max_length=20, description="Only these market keys, such as h2h")
    region: Key | None = Field(default=None, description=_REGION)
    limit: int = _limit("bookmakers", 10)


class OddsRelayEventOdds(_SpendingTool):
    """One event's prices at each bookmaker and exchange carrying it. Priced first, and refused above max_tokens."""

    name: str = "oddsrelay_event_odds"
    description: str = (
        "Reads one event's prices at each bookmaker and exchange carrying it, narrowed to the bookmakers and "
        "markets you name. Needs a key that can read raw odds. Spends tokens: priced first, and refused when the "
        "price is above this tool's cap."
    )
    args_schema: type[BaseModel] = _EventOddsInput
    narrow: ClassVar[str] = "An event is priced whole: naming fewer bookmakers or markets does not lower its price."

    def _read(self, client: OddsRelay, a: _EventOddsInput) -> dict:
        res = client.event(a.id, max_tokens=self.max_tokens, region=a.region)
        # The route takes no bookmaker or market filter, so the reply is narrowed here: the cost stays the event's.
        books = [
            {**b, "markets": [m for m in b["markets"] if m["key"] in a.markets]} if a.markets else b
            for b in res.data["bookmakers"]
            if not a.bookmakers or b["key"] in a.bookmakers
        ]
        if a.markets:  # a venue with none of the named markets is left out, so it takes no place under limit
            books = [b for b in books if b["markets"]]
        return {**res.data, **_trim(books, a.limit, "bookmakers"), "tokens": _spent(res)}


class OddsRelayToolkit(BaseToolkit):
    """The five OddsRelay tools with one key and one cap: OddsRelayToolkit(max_tokens=...).get_tools()."""

    api_key: ApiKey = _key_field()
    """Your OddsRelay API key. Left out, it is read from ODDSRELAY_KEY."""
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=0)
    """The cap handed to both tools that spend tokens: a call quoted above it is not made."""

    def get_tools(self) -> list[BaseTool]:
        key, cap = self.api_key, self.max_tokens
        return [
            OddsRelaySports(api_key=key),
            OddsRelayEvents(api_key=key),
            OddsRelayBookmakers(api_key=key),
            OddsRelayOdds(api_key=key, max_tokens=cap),
            OddsRelayEventOdds(api_key=key, max_tokens=cap),
        ]


def _json(body: Any) -> str:
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)


def _text(body: Any, key: str) -> str:
    """Compact JSON with the key hidden, raw and JSON-escaped, in case a reply or an error ever echoed it."""
    text = _json(body)
    for secret in {key, json.dumps(key)[1:-1]}:
        text = text.replace(secret, "[hidden]")
    return text


def _spent(res: Result) -> dict:
    return {"cost": res.tokens.cost, "remaining": res.tokens.remaining}


def _fit(rows: list, limit: int) -> int:
    """How many of the first `limit` rows fit in REPLY_BUDGET characters: at least one when there are any."""
    n = size = 0
    while n < min(limit, len(rows)):
        size += len(_json(rows[n]))
        if n > 0 and size > REPLY_BUDGET:
            break
        n += 1
    return n


def _trim(rows: list, limit: int, noun: str) -> dict:
    """The rows that fit, and a note counting the rest. Only the tools that spend tokens trim this way."""
    n = _fit(rows, limit)
    if n == len(rows):
        return {noun: rows}
    if n < limit:
        why = ", to keep the reply short: narrow the filters"
    else:
        why = ": raise limit (up to 100) or narrow the filters"
    return {noun: rows[:n], "more": f"{len(rows) - n} more {noun} not shown{why} (a new call, charged again)"}


def _unpack(cursor: str | None) -> tuple[int, str | None]:
    """Our cursor is "<rows shown>:<the API's cursor>", so a page cut short goes on from its first row not shown."""
    head, sep, tail = (cursor or "").partition(":")
    return (int(head), tail or None) if sep and head.isdecimal() else (0, cursor)


def _page(body: dict, skip: int, api: str | None, limit: int, noun: str) -> dict:
    """One API page from row `skip`, cut to fit. next_cursor goes on from the first row not shown."""
    rows = body["data"][skip:]
    n = _fit(rows, limit)
    if n == len(rows):
        nxt = body["meta"].get("next_cursor")
        return {noun: rows, **({"next_cursor": nxt} if nxt else {})}
    more = f"{len(rows) - n} more {noun} on this page not shown: pass next_cursor back as cursor"
    return {noun: rows[:n], "more": more, "next_cursor": f"{skip + n}:{api or ''}"}


def _explain(err: Exception, tool: _OddsRelayTool) -> dict:
    """The error the model reads: what went wrong and what the API sent with it, never a stack trace."""
    if isinstance(err, QuoteTooHigh):
        out = {
            "error": "over_max_tokens",
            "message": f"This call is priced at {err.cost} tokens, above this tool's max_tokens of {err.max_tokens}, "
            f"so it was not made and nothing was spent. {tool.narrow} Only the developer can raise the cap, where "
            "the tool is created: it is not an argument of this call.",
            "cost": err.cost,
            "max_tokens": err.max_tokens,
            "remaining": err.quote.data.get("remaining"),
        }
    elif isinstance(err, APIError):
        out = {
            "error": err.code or f"http_{err.status}",
            "status": err.status,
            "message": err.message,
            "valid": err.error.get("valid"),  # the accepted values, after an unknown key
            "cost": getattr(err, "cost", None),  # 402
            "remaining": getattr(err, "remaining", None),  # 402
            "resets_at": getattr(err, "resets_at", None),  # 402
            "retry_after": getattr(err, "retry_after", None),  # 429, after the client's own retries
            "request_id": err.request_id,
        }
    elif isinstance(err, OddsRelayError):
        out = {"error": "client_error", "message": str(err)}
    else:  # the network, a reply cut off or garbled on the way, or one this package could not read
        message = f"The call failed before a usable reply came back ({type(err).__name__}: {err})."
        if isinstance(tool, _SpendingTool):  # the paid request may have reached the API before its reply was lost
            message += " If it reached the API it may have been charged, so calling again may be charged again."
        out = {"error": "request_failed", "message": message}
    return {k: v for k, v in out.items() if v is not None}
