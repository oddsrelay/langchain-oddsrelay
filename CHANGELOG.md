# Changelog

## 0.1.0 (unreleased)

The first version, built on the `oddsrelay` client and the OddsRelay API's OpenAPI document `2026-07-08`.

- Five tools, none of which writes anything: `oddsrelay_sports`, `oddsrelay_events`, `oddsrelay_bookmakers`,
  `oddsrelay_odds` and `oddsrelay_event_odds`. `OddsRelayToolkit` makes all five with one key and one cap.
- `oddsrelay_odds` and `oddsrelay_event_odds` ask for a quote first and refuse a call priced above `max_tokens`,
  spending nothing. The cap is set in code, on the tool or the toolkit, and the model can neither see nor raise it.
- `oddsrelay_odds` takes one sport and the bookmakers you name.
- Replies cut to `limit` rows and to a size an agent can read, with a note counting the rest. On the paged tools,
  `next_cursor` goes on from the first row not shown.
- API and network errors handed back to the agent as tool errors, with the error code, `request_id` and, where the
  API sends them, the accepted keys.
- The key read from `ODDSRELAY_KEY` or `api_key`, held as a `SecretStr`, and hidden from every reply and error.
- LangChain's standard tool tests, run against stubbed replies with sockets blocked.
