# langchain-oddsrelay

LangChain tools for the OddsRelay API, built on the [`oddsrelay`](https://pypi.org/project/oddsrelay/) Python client.
They give an agent read access to matched boards, one event's prices, and the sport, event and bookmaker keys behind
them. Nothing writes. Every call that spends tokens is priced first and refused above a cap you set in code, and the
refusal tells the agent the price.

This package is made by the OddsRelay team, who run the API it reads. API docs: https://oddsrelay.io/docs

## Install

```sh
pip install langchain-oddsrelay
```

Python 3.10 and up. It installs `langchain-core` and the `oddsrelay` client with it.

## Get a key

Sign up at [app.oddsrelay.io/signup](https://app.oddsrelay.io/signup) and copy your key from the dashboard.
Free gives you 50,000 tokens for 14 days from sign-up, with no card, then no tokens until you choose a plan, on the same key. [Plans and token prices](https://oddsrelay.io/pricing).
Paid plans are for your own use. A product that serves other people runs on Enterprise.

## First call

Put your key in your environment, never in your code:

```sh
export ODDSRELAY_KEY="your key"
```

A tool runs on its own, with no model:

```python
from langchain_oddsrelay import OddsRelayOdds

odds = OddsRelayOdds()  # reads ODDSRELAY_KEY
reply = odds.invoke({
    "type": "standard",
    "sport": "soccer",  # soccer is the API's key for football
    "bookmakers": ["ladbrokes", "william_hill"],
    "markets": ["h2h"],
    "limit": 5,
})
print(reply)  # compact JSON: the board's meta, its first events, and what the call cost
```

## In an agent

`create_agent` comes from the `langchain` package. Give it any chat model that supports tool calling, and the
toolkit's tools:

```python
from langchain.agents import create_agent
from langchain_oddsrelay import OddsRelayToolkit

tools = OddsRelayToolkit().get_tools()  # reads ODDSRELAY_KEY
agent = create_agent("<provider>:<model>", tools=tools)  # your chat model
reply = agent.invoke({"messages": [{"role": "user", "content": "Which bookmaker keys does OddsRelay take?"}]})
print(reply["messages"][-1].content)
```

That question calls `oddsrelay_bookmakers`, which costs no tokens. Then ask:

> Show the OddsRelay standard board for football at Ladbrokes and William Hill, first five events.

The agent finds the keys it needs, then calls `oddsrelay_odds`, which is priced before it runs.

## Tools

| Tool | Class | What it reads | Tokens |
|---|---|---|---|
| `oddsrelay_sports` | `OddsRelaySports` | Sport and competition keys, a page at a time | none |
| `oddsrelay_events` | `OddsRelayEvents` | The events of one sport, with their ids, a page at a time | none |
| `oddsrelay_bookmakers` | `OddsRelayBookmakers` | Bookmaker and exchange keys | none |
| `oddsrelay_odds` | `OddsRelayOdds` | A matched board (`standard`, `2up`, `dutching`, `each-way`, `extra-place` or `bog`) for one sport at the bookmakers you name | priced first, capped |
| `oddsrelay_event_odds` | `OddsRelayEventOdds` | One event's prices at each bookmaker and exchange carrying it, narrowed to the bookmakers and markets you name | priced first, capped |

`OddsRelayToolkit` makes all five with one key and one cap. Each one is a LangChain `BaseTool`, so it also runs on
its own:

```python
import json

from langchain_oddsrelay import OddsRelayToolkit

sports, events, bookmakers, odds, event_odds = OddsRelayToolkit().get_tools()

print(sports.invoke({"limit": 20}))  # sport and competition keys
print(bookmakers.invoke({}))  # bookmaker and exchange keys
print(odds.invoke({"type": "each-way", "sport": "horse_racing", "bookmakers": ["ladbrokes"], "limit": 3}))

page = json.loads(events.invoke({"sport": "tennis", "limit": 5}))  # event ids for one sport
print(event_odds.invoke({"id": page["events"][0]["id"], "markets": ["h2h"]}))
```

## The token cap

`oddsrelay_odds` and `oddsrelay_event_odds` spend tokens. Before each call the tool asks the API what the call would
cost (a quote, which spends nothing). If the price is above the tool's `max_tokens`, it makes no call, spends nothing
and tells the agent the price in a tool error. A price equal to the cap goes ahead. The price is quoted a moment before
the call, and an event's price moves with the board, so an `oddsrelay_event_odds` call can cost a little more or less
than its quote.

`max_tokens` has a small default. Set your own where you make the tool or the toolkit, where `my_cap` is the highest
price, in tokens, at which a call still goes ahead:

```python
from langchain_oddsrelay import OddsRelayOdds, OddsRelayToolkit

odds = OddsRelayOdds(max_tokens=my_cap)  # one tool
tools = OddsRelayToolkit(max_tokens=my_cap).get_tools()  # both tools that spend tokens
```

The cap is not one of the tool's arguments, so the model can neither see it nor raise it: a `max_tokens` the model
sends is ignored. It caps each call, not the whole run, so an agent that makes many calls can spend up to the cap on
each. To bound a run, also limit how many calls the agent makes, for example with LangChain's
`ToolCallLimitMiddleware`.

Every reply from these two tools carries `tokens`: what the call cost and what is left. The price of an
`oddsrelay_odds` call is set by its board and, on `standard` and `dutching`, by the sport, bookmakers, exchanges and
markets it names. The kick-off window never changes it, and nor does cutting the reply to `limit` rows. When a call is
refused over the cap, the tool error says which of these can bring the price down. `oddsrelay_event_odds` is priced as
the whole event, whichever bookmakers and markets you name, and it needs a key that can read raw odds.

## Replies

Replies are compact JSON. Long lists are cut to `limit` rows (each tool has its own default) and to a size an agent
can read in one go, with a `more` note saying how many rows were left out and what to change to see them. On the two
tools that spend tokens, seeing them is a new call, charged again.
`oddsrelay_sports` and `oddsrelay_events` return `next_cursor`, which the agent passes back as `cursor`. A page cut
short goes on from the first row not shown.

Each call makes a fresh client, so a repeated call is priced and charged again.

## Errors

A refused call reaches the agent as a tool error, a `ToolMessage` with `status="error"`, and the run goes on. Its JSON
carries `error` (the API's error code), `message`, `request_id` and, where the API sends them, `valid`, the keys it
accepts: ask for an unknown bookmaker and `valid` lists the bookmaker keys. A 402 (not enough tokens left) keeps
`cost`, `remaining` and `resets_at`. A call that fails on the way, from the network or a reply cut off, is a tool
error too, `request_failed`. On the two tools that spend tokens it says the call may have been charged, since a
request can reach the API before its reply is lost. In an agent made with `create_agent`, arguments that do not fit a
tool's schema also come back to the model as an error.

To handle errors yourself, pass `handle_tool_error` as with any LangChain tool: `False` raises the `ToolException`,
and a string or a function words the error your way.

A missing key fails when the tool is made, not at its first call: a `ValueError` naming `ODDSRELAY_KEY`.

## Your key

The tools read the key from `ODDSRELAY_KEY`, or from `api_key=` where you make them, and hold it as a `SecretStr`,
so a repr or a log line shows `**********`. It travels only in the `Authorization` header to
`https://api.oddsrelay.io`, never in a URL. If a reply or an error ever echoed it, the agent would see `[hidden]`
instead.

## Async

`ainvoke` works, and so do agents that call several tools at once. The `oddsrelay` client is synchronous, so each
async call runs in a worker thread, and each call makes its own client, so calls share nothing.

## Develop

```sh
python -m pip install -e . --group test  # pip 25.1 or later
python -m pytest
```

The tests include LangChain's standard tool tests (`langchain-tests`). They run the real client against stubbed
replies built from the API's published examples, with sockets blocked, so no test reaches the network.

## More from OddsRelay

- [oddsrelay-python](https://github.com/oddsrelay/oddsrelay-python): the Python client these tools are built on, `pip install oddsrelay`
- [oddsrelay-mcp](https://github.com/oddsrelay/oddsrelay-mcp): an MCP server that gives Claude, Cursor and other AI assistants read access, with a token cap
- [oddsrelay-js](https://github.com/oddsrelay/oddsrelay-js): the JavaScript and TypeScript client, `npm install oddsrelay`
- [oddsrelay-r](https://github.com/oddsrelay/oddsrelay-r): the R client
- [oddsrelay-sheets](https://github.com/oddsrelay/oddsrelay-sheets): formulas and a template spreadsheet for Google Sheets
- [examples](https://github.com/oddsrelay/examples): short programs to copy, one folder each
- [Libraries and tools](https://oddsrelay.io/docs/libraries), the [docs](https://oddsrelay.io/docs), the [API reference](https://oddsrelay.io/docs/reference) and the [OpenAPI file](https://api.oddsrelay.io/v2/openapi.json)
- Contact: hello@oddsrelay.io

## Licence

MIT. See [LICENSE](https://github.com/oddsrelay/langchain-oddsrelay/blob/main/LICENSE).
