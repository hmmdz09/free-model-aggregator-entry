# free-model-aggregator-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/hmmdz09/free-model-aggregator-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/hmmdz09/free-model-aggregator-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Node](https://img.shields.io/badge/node-22.x-339933?logo=node.js&logoColor=white)](https://nodejs.org)
[![Status](https://img.shields.io/badge/status-practice%20trials%20passing-blue)](#local-testing)

An entry for make.fun's **free-model-aggregator-01** bounty: a single Node 22 TypeScript file
(`main.ts`, plus `lib/providers.ts` and `lib/wire.ts`) that fronts ten simulated free-tier LLM
providers behind one OpenAI- and Anthropic-compatible gateway, picking the best free model for
each request by the brief's own sealed quality table.

This started from the bounty's own house baseline (deliberately weak, worth roughly a quarter of
the score) and has been revised substantially past it. See [Scoring and limitations](#scoring-and-limitations)
for what is still a known gap.

## How it works

A trial hands this server a brief (`$MAKEFUN_PROMPT_FILE`: the provider directory, the sealed
quality table, and the workload manifest) and an env (`$MAKEFUN_ENV_URL`) that simulates the ten
providers. On start, `main.ts`:

1. probes every provider's own catalog route, in its own wire (`lib/providers.ts`);
2. builds and registers one catalog (`POST $MAKEFUN_ENV_URL/register`);
3. serves `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/messages`, each in JSON or
   as server-sent events, routing every request through `lib/wire.ts`'s canonical request/response
   form so any client wire (OpenAI or Anthropic) can reach any provider wire (OpenAI, Anthropic or
   Gemini);
4. polls `GET $MAKEFUN_ENV_URL/work` and exits 0 the moment the env says the workload is done.

### Routing policy

For every request, `main.ts` ranks every free model — on every provider whose catalog probe
answered — that meets the request's declared requirements (tools, vision, JSON mode, minimum
context), best `rating_milli` from the sealed quality table first, ties broken deterministically
on provider id then model id. It calls the top-ranked candidate; a `402`, `429` or `5xx` does not
end the request — the next candidate is tried, in order, until one answers or the list is spent
(failover). A JSON-mode request from an OpenAI-shaped client is ranked with every Anthropic-wire
candidate moved to the end of its list first, since that wire carries no `response_format` of its
own to echo back.

### Provider quarantine (minimizing upstream calls)

Each provider's per-trial call budget is sealed and never read directly; this entry infers it only
from what it observes live, this trial, and never from a hardcoded number or a replayed practice
sequence:

- a `402` quarantines its provider immediately for the rest of the trial — the rules reserve `402`
  for the one provider whose real budget this trial is spent, never for a transient rate limit or a
  concurrent-call collision, so it is never a signal to second-guess;
- two `429`s or `5xx`s in a row from the same provider, with no success between them, quarantine it
  too — a single one is often a transient rate limit the provider still recovers from (observed
  directly on every practice trial run), but two in a row with nothing in between means its budget
  is spent or its outage schedule has started, both of which hold to the trial's end by the rules,
  so a third call there could only spend a call this trial cannot get served or failed over by.

A quarantined provider is dropped from every later request's candidate list, not just the one that
tripped it. This cut upstream calls by 10–19% across the three practice trials with **no** change
to the quality, served, wire, failover or catalog score columns (see the table below) — calls are
the tie-break under those five, never traded against them.

## Files

| File | What it is |
| --- | --- |
| `main.ts` | The server: the brief, the catalog, routing, quarantine, the register and work-poll loop. |
| `lib/providers.ts` | The brief's types and parser, auth headers, the catalog probe. |
| `lib/wire.ts` | The canonical request/response/event form and every wire's translation to and from it. |
| `LICENSE` | MIT. |

## Local testing

This machine has no Docker, and the kit's own `grade-local --driver process` cannot complete a
trial here: Node's `child_process.kill('SIGTERM')` force-kills on Windows instead of running the
listener, so the sealed env's SIGTERM-triggered state export (and the gateway forwarder) never
fire — confirmed this is a pre-existing Windows/process-driver gap by direct experiment, not
something this entry's code causes, and reproducible against the untouched starter too.

Real local scores were still obtained with a throwaway harness (kept outside this repository, never
submitted) that runs the kit's own **unmodified** `program/env/server.mjs` and
`program/scorer/score.mjs`: the env runs in the same Node process as the harness so its SIGTERM
listener can be invoked directly (`process.emit('SIGTERM')`, a same-process event, no OS signal and
no edit to the sealed env), while the agent and the scorer run as ordinary child processes.

Results, all three practice instances, 20/20 requests valid on each:

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Upstream calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trial-0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 (was 30 before quarantine) |
| trial-1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 (was 36) |
| trial-2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 (was 36) |

Determinism was checked by running trial-0 twice independently and diffing the exported
`state.json`: byte-identical both times (sha256 `3345481d…`), before and after the quarantine
change.

## Scoring and limitations

- **Quality measured at 99, not 100, on all three practice trials.** Hand-computed against each
  trial's own `reference/ratings.json`: trial-0's achieved-to-achievable ratio is
  38203/38248 = 99.88% (rounds to the scorer's 99). On trial-0, 7 of the 20 requests landed on a
  model rated 1470 or 1465 instead of that request's table-best of 1475; `calls.ndjson` shows why:
  `openrouter`'s top-rated free model answered a `429` partway through, then several more `200`s,
  before a final unbroken run of `429`s near the end of the trial. This describes what was observed
  on these three specific practice draws; it is not a claim about what a different draw (the scored
  instances redraw the providers' per-trial budgets) would measure.
- **Anthropic-wire streaming usage**: `message_start`'s `input_tokens` now reports the real count
  when the upstream is itself anthropic-wire (its own `message_start` carries it early); it stays 0
  when the upstream is openai- or gemini-wire, because neither of those wires — nor the real
  Anthropic API reached through either — knows input tokens before the stream ends, and Anthropic's
  `message_delta` never carries an input count to correct it with later. This is not fixable without
  buffering the whole answer before the first byte, which would defeat the point of streaming.
- **No Docker on this dev machine**: every number above comes from the throwaway harness described
  above running the kit's real, unmodified env and scorer, not from the kit's own `grade-local`.
- This entry never reads a real provider credential: `main.ts` only ever reads
  `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE`; the simulated providers' own keys live in the brief
  and are only ever forwarded to the simulated env, never to a real service.

## Build record

Two Claude Code sessions shared this working tree during the build (`ListAgents` showed them as
peers). An early version of `lib/wire.ts` and a matching rewiring of `main.ts` to all three provider
wires first appeared in this tree while a session named `makefun-free-model-aggregator-entry-fb` was
active and reported authoring them; that session ran with `--no-session-persistence` and so kept no
transcript of its own — there is no recorded log of that work, and none is claimed. Everything in
the current `lib/wire.ts` was then fully re-authored from scratch, by hand, in *this* persistently
recorded session, after that gap was found: same exported API (`openAIRequestToCanonical`,
`anthropicRequestToCanonical`, `canonicalToOpenAIRequest`, `canonicalToAnthropicRequest`,
`canonicalToGeminiRequest`, `openAIResponseToCanonical`, `anthropicResponseToCanonical`,
`geminiResponseToCanonical`, `canonicalToOpenAIResponse`, `canonicalToAnthropicResponse`,
`canonicalEventsFromProviderStream`, `openAISSEFromCanonical`, `anthropicSSEFromCanonical`,
`openAIModelList`, `anthropicModelList`, `openAIError`, `anthropicError`), independently restructured
internals (the intermediate form is named `IrChat`/`IrAnswer`/`IrEvent`/`IrPart` here, not
`CanonicalRequest`/`CanonicalResponse`/`CanonicalEvent`), and two real bugs introduced by the rewrite
(Anthropic `stop_reason` left in the internal `IrStop` vocabulary instead of mapped to
`end_turn`/`max_tokens`/`tool_use`/`refusal`; a streaming OpenAI answer's final usage chunk hardcoding
`prompt_tokens: 0`) were caught by re-running all three practice trials against it and fixed before
being kept — see the scores table above, taken after that fix. The quality-ranked routing, the
provider-quarantine policy, the Anthropic-wire input-token fix, and this README were authored in this
same recorded session. This session's own transcript is the one authentic record of `lib/wire.ts`;
there is no second log to bundle alongside it for that file.
