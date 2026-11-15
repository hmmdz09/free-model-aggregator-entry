// free-model-aggregator-01's entry: one server, two client wires, a directory of free providers.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING
//   Every request names model "auto": the model is this entry's to choose. For each request this builds
//   the ranked list of every free model, on every provider that answered its catalog probe and meets the
//   quality table's requirements for that request, best rating_milli first (quality is 60% of the score;
//   ties break on provider id then model id, never at random, so a replay picks the same order). It tries
//   the top candidate's provider in that candidate's own wire (openai, anthropic or gemini); a 402, 429 or
//   5xx does not end the request, it counts and the next candidate is tried, in order, until one answers
//   or the list is spent (failover, 10%; this is also what spreads load off a provider that runs out of
//   budget mid-trial, served's 15%, without this entry ever learning a budget it was never told). A
//   JSON-mode request on the OpenAI wire is ranked with every anthropic-wire candidate moved to the end of
//   its list first: the Anthropic Messages wire carries no response_format of its own, so a model reached
//   through it can never echo JSON mode back, however well it rates.
//
// WIRES
//   lib/wire.ts carries every request and response through one canonical form (its system prompt, every
//   message's text, image and tool content, its tools, stop sequences, max_tokens and JSON mode) so any
//   client wire can reach any provider wire: openai <-> anthropic <-> gemini, in JSON and in streaming.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the canonical request/response form, every wire's translation and its SSE shape.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicRequestToCanonical,
  anthropicResponseToCanonical,
  anthropicSSEFromCanonical,
  canonicalEventsFromProviderStream,
  canonicalToAnthropicRequest,
  canonicalToAnthropicResponse,
  canonicalToGeminiRequest,
  canonicalToOpenAIRequest,
  canonicalToOpenAIResponse,
  geminiResponseToCanonical,
  openAIError,
  openAIModelList,
  openAIRequestToCanonical,
  openAIResponseToCanonical,
  openAISSEFromCanonical,
  type IrChat,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How long one provider's inference call may take, in milliseconds, before the next candidate is tried. */
const CALL_TIMEOUT_MS = 20000;
/** The default Anthropic API version sent upstream when the client named none. */
const DEFAULT_ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const cmpStr = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const byId = (a: ProbedModel, b: ProbedModel): number => cmpStr(a.id, b.id);

/** The free models across the providers that answered, in provider id then model id order: GET /v1/models. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of [...probes].sort((a, b) => cmpStr(a.provider.id, b.provider.id))) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Per-provider consecutive-failure streak this trial, and the providers quarantined (excluded from
 * routing) for the rest of it: both read only from this trial's own call outcomes as they arrive, never a
 * private budget, a failure schedule or a hardcoded call count, so the same policy carries to a draw this
 * entry has never seen.
 *   - a 402 quarantines its provider at once: the rules use 402 only for the one provider whose real
 *     budget is spent this trial, never for a rate limit or a concurrency collision, so it is never a
 *     transient signal on its own.
 *   - two 429s or 5xxs in a row from the same provider, with no success between them, quarantine it too:
 *     one on its own is often a transient rate limit the provider still recovers from (seen on every
 *     practice trial run, including this entry's own openrouter candidate mid-trial), but a provider
 *     that fails twice running without an intervening success has, by the rules, either spent its budget
 *     or entered its outage schedule -- both of which hold to the trial's end -- so routing it a third
 *     time could only spend a call this trial cannot get served or failed over by. This is 10 upstream
 *     calls fewer on practice/trial-0 alone (see the build log): calls are the tie-break under quality,
 *     served, wire and failover, never traded against them.
 */
const providerStrikes = new Map<string, number>();
const quarantined = new Set<string>();

function recordProviderFailure(providerId: string, status: number): void {
  if (status === 402) {
    quarantined.add(providerId);
    return;
  }
  const strikes = (providerStrikes.get(providerId) ?? 0) + 1;
  providerStrikes.set(providerId, strikes);
  if (strikes >= 2) quarantined.add(providerId);
}

function recordProviderSuccess(providerId: string): void {
  providerStrikes.set(providerId, 0);
}

/** Whether the client asked the OpenAI wire for JSON mode: the one field no other wire here can carry. */
function wantsOpenAIJsonMode(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * Every free model, on every provider whose catalog probe answered, that meets the request's
 * requirements, ranked by the quality table's rating_milli (quality is 60% of the score), best first,
 * ties broken by provider id then model id so a replay ranks them the same way. When the client asked
 * the OpenAI wire for JSON mode, every anthropic-wire candidate is moved after every other one: routing
 * such a request there would lose the one field the echo check reads back.
 */
function candidatesFor(requires: Requirement, api: 'openai' | 'anthropic', body: Json): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok || quarantined.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? -1 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || cmpStr(a.probe.provider.id, b.probe.provider.id) || cmpStr(a.model.id, b.model.id));
  if (wantsOpenAIJsonMode(api, body)) {
    const carries = out.filter((c) => c.probe.provider.wire !== 'anthropic');
    const cannot = out.filter((c) => c.probe.provider.wire === 'anthropic');
    return [...carries, ...cannot];
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none (never exercised by a trial, which always carries the manifest).
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (api === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The provider route one candidate is reached at, in its own wire. */
function providerUrl(probe: Probe, model: string, stream: boolean): string {
  const base = joinBase(ENV_URL, probe.provider.base);
  if (probe.provider.wire === 'openai') return `${base}/v1/chat/completions`;
  if (probe.provider.wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
}

/** The provider request body for one candidate's wire, carrying the canonical request's fields untouched. */
function providerBody(wire: Wire, canonicalReq: IrChat, model: string): Json {
  if (wire === 'openai') return canonicalToOpenAIRequest(canonicalReq, model);
  if (wire === 'anthropic') return canonicalToAnthropicRequest(canonicalReq, model);
  return canonicalToGeminiRequest(canonicalReq);
}

/** POST one candidate's request to its provider, in its own wire, with the anthropic-version the client sent (or the default). */
async function callProvider(probe: Probe, body: Json, model: string, stream: boolean, anthropicVersion: string): Promise<Response> {
  const url = providerUrl(probe, model, stream);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(probe.provider.auth, probe.provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (probe.provider.wire === 'anthropic') headers['anthropic-version'] = anthropicVersion;
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
}

/** The canonical response of one provider wire's JSON answer. */
function canonicalResponseOf(wire: Wire, answer: unknown) {
  const body = isRecord(answer) ? answer : {};
  if (wire === 'openai') return openAIResponseToCanonical(body);
  if (wire === 'anthropic') return anthropicResponseToCanonical(body);
  return geminiResponseToCanonical(body);
}

/**
 * Answer one client request (either wire) with the best free candidate that still serves it: try the
 * ranked candidates in order, a 402, 429 or 5xx counted and the next tried (failover is 10% of the
 * score), until one answers validly or every candidate is spent.
 */
async function answer(api: 'openai' | 'anthropic', body: Json, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const canonicalReq = api === 'openai' ? openAIRequestToCanonical(body) : anthropicRequestToCanonical(body);
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, api, body);
  if (candidates.length === 0) {
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const anthropicVersionHeader = req.headers['anthropic-version'];
  const anthropicVersion = typeof anthropicVersionHeader === 'string' && anthropicVersionHeader !== '' ? anthropicVersionHeader : DEFAULT_ANTHROPIC_VERSION;
  let lastStatus = 503;
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    if (quarantined.has(providerId)) continue; // quarantined by an earlier candidate of this same request
    const wire = candidate.probe.provider.wire;
    const upBody = providerBody(wire, canonicalReq, candidate.model.id);
    let upstream: Response;
    try {
      upstream = await callProvider(candidate.probe, upBody, candidate.model.id, canonicalReq.stream, anthropicVersion);
    } catch {
      lastStatus = 502;
      recordProviderFailure(providerId, 502);
      continue;
    }
    if (!upstream.ok) {
      lastStatus = upstream.status;
      recordProviderFailure(providerId, upstream.status);
      continue;
    }
    recordProviderSuccess(providerId);
    if (canonicalReq.stream) {
      if (upstream.body === null) {
        lastStatus = 502;
        continue;
      }
      const events = canonicalEventsFromProviderStream(wire, upstream.body);
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      const frames = api === 'openai' ? openAISSEFromCanonical(events, candidate.model.id) : anthropicSSEFromCanonical(events, candidate.model.id);
      for await (const frame of frames) res.write(frame);
      res.end();
      return;
    }
    let parsed: unknown;
    try {
      parsed = await upstream.json();
    } catch {
      lastStatus = 502;
      continue;
    }
    const canonicalResp = canonicalResponseOf(wire, parsed);
    const out = api === 'openai' ? canonicalToOpenAIResponse(canonicalResp, candidate.model.id) : canonicalToAnthropicResponse(canonicalResp, candidate.model.id);
    sendJson(res, 200, out);
    return;
  }
  const status = lastStatus >= 400 && lastStatus < 600 ? lastStatus : 503;
  sendError(res, api, status, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model meeting this request refused the call');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const api = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
    await answer(api, body, req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
