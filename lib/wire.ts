// The client wires (OpenAI Chat Completions, Anthropic Messages) and the three provider wires (openai,
// anthropic, gemini), bridged through one small intermediate form instead of six hand-written
// request/response pairings.
//
// A trial's client speaks to this entry as either an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), JSON or server-sent events, and main.ts may need to answer with a
// model reached through any of the three provider wires. Writing openai->anthropic, openai->gemini,
// anthropic->openai, anthropic->gemini, anthropic->anthropic and so on by hand is six translations (twelve
// once responses are counted); instead every request and response passes through one intermediate shape
// (prefixed `Ir` below, for "intermediate representation") that keeps exactly what a request needs carried
// through untouched -- the system prompt, every turn's text/image/tool content, the tool definitions, the
// stop sequences, the output cap and JSON mode -- because the scorer's checks compare what a provider
// echoes against what the *client* itself sent, not anything carried only through this file.
//
//   client body -> IrChat      (readOpenAIChat / readAnthropicChat)
//   IrChat -> provider body    (irChatToOpenAI / irChatToAnthropic / irChatToGemini)
//   provider body -> IrAnswer  (openAIAnswerToIr / anthropicAnswerToIr / geminiAnswerToIr)
//   IrAnswer -> client body    (irAnswerToOpenAI / irAnswerToAnthropic)
//
// and the streaming case the same way, with one event stream in between (IrEvent, parsed per provider wire
// by readProviderStream, rendered per client wire by writeOpenAIStream / writeAnthropicStream).
//
// Nothing here reads the clock or anything unseeded: the same request body and the same provider answer
// must produce the same bytes on a replay, so the two model-list builders below take a fixed timestamp
// rather than the real one.

import { isRecord, type Json } from './providers.ts';

// ---------------------------------------------------------------------------------------------------------
// GET /v1/models and the two wires' error envelopes.
// ---------------------------------------------------------------------------------------------------------

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_LIST_EPOCH_SECONDS = 1767225600;
const FIXED_LIST_TIMESTAMP = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request (an `Authorization` header). */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return { object: 'list', data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_LIST_EPOCH_SECONDS, owned_by: m.provider })) };
}

/** GET /v1/models on an Anthropic-shaped request (an `x-api-key` header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return { data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_LIST_TIMESTAMP })), has_more: false };
}

/** An error in the Anthropic Messages envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in the OpenAI Chat Completions envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------------------------------------------------
// The intermediate chat request (Ir = intermediate representation).
// ---------------------------------------------------------------------------------------------------------

/** One piece of a turn's content, in whichever client wire it arrived. */
export type IrPart =
  | { kind: 'text'; text: string }
  | { kind: 'image_b64'; mime: string; base64: string }
  | { kind: 'image_url'; url: string }
  | { kind: 'tool_call'; callId: string; toolName: string; args: Json }
  | { kind: 'tool_result'; callId: string; text: string };

export interface IrTurn {
  speaker: 'user' | 'assistant';
  parts: IrPart[];
}

export interface IrToolDef {
  name: string;
  description: string;
  schema: Json;
}

/** How the client asked the model to use its tools; absent means the provider's own default. */
export type IrToolPolicy = { force: 'auto' } | { force: 'none' } | { force: 'any' } | { force: 'named'; name: string };

export interface IrChat {
  system: string;
  turns: IrTurn[];
  maxOutputTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stopSequences: string[];
  tools: IrToolDef[];
  toolPolicy: IrToolPolicy | null;
  wantsJson: boolean;
  stream: boolean;
}

/** How the answer stopped, independent of any one wire's own vocabulary for it. */
export type IrStop = 'done' | 'length' | 'tool_call' | 'blocked';

export interface IrAnswer {
  parts: IrPart[];
  stop: IrStop;
  inputTokens: number;
  outputTokens: number;
}

/** One streamed increment of an answer, independent of any one wire's own event shape. */
export type IrEvent =
  | { kind: 'input_tokens'; count: number }
  | { kind: 'text_delta'; text: string }
  | { kind: 'tool_call_start'; slot: number; callId: string; toolName: string }
  | { kind: 'tool_call_delta'; slot: number; argsJsonFragment: string }
  | { kind: 'finished'; stop: IrStop; outputTokens: number | null };

// ---------------------------------------------------------------------------------------------------------
// Small shared helpers.
// ---------------------------------------------------------------------------------------------------------

function asNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asPositiveInt(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : 0;
}

function asStringArray(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** Every text block of a string-or-block-array content value, joined; '' when there is none. */
function flattenText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const pieces: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') pieces.push(block.text);
  return pieces.join('\n');
}

/** A tool result's content rendered as plain text for the one wire (Chat Completions) that wants that. */
function toolResultAsText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = flattenText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's argument payload as an object, whether the provider sent it as a string or already parsed. */
function toolArgsAsObject(raw: unknown): Json {
  if (isRecord(raw)) return raw;
  if (typeof raw === 'string' && raw.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(raw);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not parseable JSON: treat it as no arguments rather than fail the whole answer over it.
    }
  }
  return {};
}

/** Every wire's own finish/stop vocabulary, folded onto IrStop. */
function stopOf(nativeReason: unknown): IrStop {
  if (nativeReason === 'length' || nativeReason === 'max_tokens' || nativeReason === 'MAX_TOKENS') return 'length';
  if (nativeReason === 'tool_calls' || nativeReason === 'function_call' || nativeReason === 'tool_use') return 'tool_call';
  if (nativeReason === 'content_filter' || nativeReason === 'refusal' || nativeReason === 'SAFETY') return 'blocked';
  return 'done';
}

// ---------------------------------------------------------------------------------------------------------
// Client body -> IrChat.
// ---------------------------------------------------------------------------------------------------------

/** An OpenAI Chat Completions request body as an IrChat. */
export function openAIRequestToCanonical(body: Json): IrChat {
  const turns: IrTurn[] = [];
  const systemPieces: string[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = flattenText(raw.content);
      if (text !== '') systemPieces.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      turns.push({ speaker: 'user', parts: [{ kind: 'tool_result', callId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', text: toolResultAsText(raw.content) }] });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: IrPart[] = [];
      const text = flattenText(raw.content);
      if (text !== '') parts.push({ kind: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          kind: 'tool_call',
          callId: typeof call.id === 'string' ? call.id : '',
          toolName: typeof call.function.name === 'string' ? call.function.name : '',
          args: toolArgsAsObject(call.function.arguments),
        });
      }
      turns.push({ speaker: 'assistant', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      turns.push({ speaker: 'user', parts: [{ kind: 'text', text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: IrPart[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') parts.push({ kind: 'image_url', url: block.image_url.url });
    }
    turns.push({ speaker: 'user', parts });
  }

  const tools: IrToolDef[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
  }

  let toolPolicy: IrToolPolicy | null = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolPolicy = { force: 'auto' };
  else if (choice === 'none') toolPolicy = { force: 'none' };
  else if (choice === 'required') toolPolicy = { force: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') toolPolicy = { force: 'named', name: choice.function.name };

  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemPieces.join('\n'),
    turns,
    maxOutputTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    stopSequences: asStringArray(body.stop),
    tools,
    toolPolicy,
    wantsJson: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages request body as an IrChat. */
export function anthropicRequestToCanonical(body: Json): IrChat {
  const turns: IrTurn[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const speaker: IrTurn['speaker'] = raw.role === 'assistant' ? 'assistant' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      turns.push({ speaker, parts: [{ kind: 'text', text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: IrPart[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ kind: 'text', text: block.text });
      } else if (block.type === 'tool_use') {
        parts.push({ kind: 'tool_call', callId: typeof block.id === 'string' ? block.id : '', toolName: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} });
      } else if (block.type === 'tool_result') {
        parts.push({ kind: 'tool_result', callId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', text: toolResultAsText(block.content) });
      } else if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') parts.push({ kind: 'image_b64', mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', base64: source.data });
        else if (source.type === 'url' && typeof source.url === 'string') parts.push({ kind: 'image_url', url: source.url });
      }
    }
    turns.push({ speaker, parts });
  }

  const tools: IrToolDef[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
  }

  let toolPolicy: IrToolPolicy | null = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolPolicy = { force: 'auto' };
    else if (choice.type === 'none') toolPolicy = { force: 'none' };
    else if (choice.type === 'any') toolPolicy = { force: 'any' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolPolicy = { force: 'named', name: choice.name };
  }

  return {
    system: flattenText(body.system),
    turns,
    maxOutputTokens: asNumber(body.max_tokens),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    stopSequences: asStringArray(body.stop_sequences),
    tools,
    toolPolicy,
    wantsJson: false, // the Anthropic wire carries no JSON-mode field of its own
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------------------------------------
// IrChat -> provider body.
// ---------------------------------------------------------------------------------------------------------

/** IrChat as an OpenAI Chat Completions request body, naming `model`. */
export function canonicalToOpenAIRequest(chat: IrChat, model: string): Json {
  const messages: Json[] = [];
  if (chat.system !== '') messages.push({ role: 'system', content: chat.system });
  for (const turn of chat.turns) {
    const results = turn.parts.filter((p): p is Extract<IrPart, { kind: 'tool_result' }> => p.kind === 'tool_result');
    for (const r of results) messages.push({ role: 'tool', tool_call_id: r.callId, content: r.text });
    const rest = turn.parts.filter((p) => p.kind !== 'tool_result');
    if (rest.length === 0) continue;
    if (turn.speaker === 'assistant') {
      const calls = rest.filter((p): p is Extract<IrPart, { kind: 'tool_call' }> => p.kind === 'tool_call');
      const text = rest
        .filter((p): p is Extract<IrPart, { kind: 'text' }> => p.kind === 'text')
        .map((p) => p.text)
        .join('\n');
      const message: Json = { role: 'assistant', content: text !== '' ? text : calls.length > 0 ? null : '' };
      if (calls.length > 0) message.tool_calls = calls.map((c) => ({ id: c.callId, type: 'function', function: { name: c.toolName, arguments: JSON.stringify(c.args) } }));
      messages.push(message);
      continue;
    }
    const parts: Json[] = [];
    for (const part of rest) {
      if (part.kind === 'text') parts.push({ type: 'text', text: part.text });
      else if (part.kind === 'image_b64') parts.push({ type: 'image_url', image_url: { url: `data:${part.mime};base64,${part.base64}` } });
      else if (part.kind === 'image_url') parts.push({ type: 'image_url', image_url: { url: part.url } });
    }
    if (parts.length === 1 && parts[0]?.type === 'text') messages.push({ role: 'user', content: parts[0].text });
    else if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (chat.maxOutputTokens !== null) out.max_tokens = chat.maxOutputTokens;
  if (chat.temperature !== null) out.temperature = chat.temperature;
  if (chat.topP !== null) out.top_p = chat.topP;
  if (chat.stopSequences.length > 0) out.stop = chat.stopSequences;
  if (chat.tools.length > 0) out.tools = chat.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  if (chat.toolPolicy !== null) {
    const p = chat.toolPolicy;
    out.tool_choice = p.force === 'named' ? { type: 'function', function: { name: p.name } } : p.force === 'any' ? 'required' : p.force;
  }
  if (chat.wantsJson) out.response_format = { type: 'json_object' };
  if (chat.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** IrChat as an Anthropic Messages request body, naming `model`. */
export function canonicalToAnthropicRequest(chat: IrChat, model: string): Json {
  const messages: Json[] = [];
  for (const turn of chat.turns) {
    const blocks: Json[] = [];
    for (const part of turn.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'tool_call') blocks.push({ type: 'tool_use', id: part.callId, name: part.toolName, input: part.args });
      else if (part.kind === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: part.callId, content: part.text });
      else if (part.kind === 'image_b64') blocks.push({ type: 'image', source: { type: 'base64', media_type: part.mime, data: part.base64 } });
      else if (part.kind === 'image_url') blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
    }
    if (blocks.length === 1 && blocks[0]?.type === 'text') messages.push({ role: turn.speaker, content: blocks[0].text });
    else if (blocks.length > 0) messages.push({ role: turn.speaker, content: blocks });
  }

  const out: Json = { model, messages };
  if (chat.system !== '') out.system = chat.system;
  if (chat.maxOutputTokens !== null) out.max_tokens = chat.maxOutputTokens;
  if (chat.temperature !== null) out.temperature = chat.temperature;
  if (chat.topP !== null) out.top_p = chat.topP;
  if (chat.stopSequences.length > 0) out.stop_sequences = chat.stopSequences;
  if (chat.tools.length > 0) out.tools = chat.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  if (chat.toolPolicy !== null) {
    const p = chat.toolPolicy;
    out.tool_choice = p.force === 'named' ? { type: 'tool', name: p.name } : p.force === 'any' ? { type: 'any' } : p.force === 'none' ? { type: 'none' } : { type: 'auto' };
  }
  if (chat.stream) out.stream = true;
  return out;
}

/** IrChat as a Gemini generateContent request body (the model rides the route, not this body). */
export function canonicalToGeminiRequest(chat: IrChat): Json {
  const contents: Json[] = [];
  for (const turn of chat.turns) {
    const parts: Json[] = [];
    for (const part of turn.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'tool_call') parts.push({ functionCall: { name: part.toolName, args: part.args } });
      else if (part.kind === 'tool_result') parts.push({ functionResponse: { name: part.callId, response: { content: part.text } } });
      else if (part.kind === 'image_b64') parts.push({ inlineData: { mimeType: part.mime, data: part.base64 } });
      else if (part.kind === 'image_url') parts.push({ fileData: { mimeType: 'application/octet-stream', fileUri: part.url } });
    }
    if (parts.length > 0) contents.push({ role: turn.speaker === 'assistant' ? 'model' : 'user', parts });
  }

  const generationConfig: Json = {};
  if (chat.maxOutputTokens !== null) generationConfig.maxOutputTokens = chat.maxOutputTokens;
  if (chat.temperature !== null) generationConfig.temperature = chat.temperature;
  if (chat.topP !== null) generationConfig.topP = chat.topP;
  if (chat.stopSequences.length > 0) generationConfig.stopSequences = chat.stopSequences;
  if (chat.wantsJson) generationConfig.responseMimeType = 'application/json';

  const out: Json = { contents };
  if (chat.system !== '') out.systemInstruction = { parts: [{ text: chat.system }] };
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  if (chat.tools.length > 0) out.tools = [{ functionDeclarations: chat.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  if (chat.toolPolicy !== null) {
    const p = chat.toolPolicy;
    const mode = p.force === 'named' || p.force === 'any' ? 'ANY' : p.force === 'none' ? 'NONE' : 'AUTO';
    out.toolConfig = { functionCallingConfig: { mode, ...(p.force === 'named' ? { allowedFunctionNames: [p.name] } : {}) } };
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------
// Provider body -> IrAnswer (one-shot JSON answers).
// ---------------------------------------------------------------------------------------------------------

/** An OpenAI Chat Completions answer as an IrAnswer. */
export function openAIResponseToCanonical(body: Json): IrAnswer {
  const choice = Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const parts: IrPart[] = [];
  const text = typeof message.content === 'string' ? message.content : flattenText(message.content);
  if (text !== '') parts.push({ kind: 'text', text });
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    parts.push({ kind: 'tool_call', callId: typeof call.id === 'string' ? call.id : 'toolu_unknown', toolName: typeof call.function.name === 'string' ? call.function.name : '', args: toolArgsAsObject(call.function.arguments) });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return { parts, stop: stopOf(choice.finish_reason), inputTokens: asPositiveInt(usage.prompt_tokens), outputTokens: asPositiveInt(usage.completion_tokens) };
}

/** An Anthropic Messages answer as an IrAnswer. */
export function anthropicResponseToCanonical(body: Json): IrAnswer {
  const parts: IrPart[] = [];
  for (const block of Array.isArray(body.content) ? body.content : []) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'tool_use') parts.push({ kind: 'tool_call', callId: typeof block.id === 'string' ? block.id : 'toolu_unknown', toolName: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return { parts, stop: stopOf(body.stop_reason), inputTokens: asPositiveInt(usage.input_tokens), outputTokens: asPositiveInt(usage.output_tokens) };
}

/** A Gemini generateContent answer as an IrAnswer. */
export function geminiResponseToCanonical(body: Json): IrAnswer {
  const candidate = Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts: IrPart[] = [];
  for (const part of Array.isArray(content.parts) ? content.parts : []) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string' && part.text !== '') parts.push({ kind: 'text', text: part.text });
    else if (isRecord(part.functionCall) && typeof part.functionCall.name === 'string') parts.push({ kind: 'tool_call', callId: `toolu_${part.functionCall.name}`, toolName: part.functionCall.name, args: isRecord(part.functionCall.args) ? part.functionCall.args : {} });
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { parts, stop: stopOf(candidate.finishReason), inputTokens: asPositiveInt(usage.promptTokenCount), outputTokens: asPositiveInt(usage.candidatesTokenCount) };
}

// ---------------------------------------------------------------------------------------------------------
// IrAnswer -> client body (one-shot JSON answers).
// ---------------------------------------------------------------------------------------------------------

function splitTextAndCalls(parts: readonly IrPart[]): { text: string; calls: Extract<IrPart, { kind: 'tool_call' }>[] } {
  const text = parts
    .filter((p): p is Extract<IrPart, { kind: 'text' }> => p.kind === 'text')
    .map((p) => p.text)
    .join('');
  const calls = parts.filter((p): p is Extract<IrPart, { kind: 'tool_call' }> => p.kind === 'tool_call');
  return { text, calls };
}

const OPENAI_FINISH_REASON: Record<IrStop, string> = { done: 'stop', length: 'length', tool_call: 'tool_calls', blocked: 'content_filter' };
const ANTHROPIC_STOP_REASON: Record<IrStop, string> = { done: 'end_turn', length: 'max_tokens', tool_call: 'tool_use', blocked: 'refusal' };

/** An IrAnswer as an OpenAI Chat Completions answer, naming `model` as the client itself asked for it. */
export function canonicalToOpenAIResponse(answer: IrAnswer, model: string): Json {
  const { text, calls } = splitTextAndCalls(answer.parts);
  const message: Json = { role: 'assistant', content: text !== '' ? text : calls.length > 0 ? null : '' };
  if (calls.length > 0) message.tool_calls = calls.map((c) => ({ id: c.callId, type: 'function', function: { name: c.toolName, arguments: JSON.stringify(c.args) } }));
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_LIST_EPOCH_SECONDS,
    model,
    choices: [{ index: 0, message, finish_reason: OPENAI_FINISH_REASON[answer.stop], logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** An IrAnswer as an Anthropic Messages answer, naming `model` as the client itself asked for it. */
export function canonicalToAnthropicResponse(answer: IrAnswer, model: string): Json {
  const { text, calls } = splitTextAndCalls(answer.parts);
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const c of calls) content.push({ type: 'tool_use', id: c.callId, name: c.toolName, input: c.args });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: ANTHROPIC_STOP_REASON[answer.stop],
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

// ---------------------------------------------------------------------------------------------------------
// Server-sent events: the frame reader every provider stream shares, then each wire's own parser.
// ---------------------------------------------------------------------------------------------------------

/** One SSE event: `event: <name>` when named, a `data:` line, then the blank line that ends the frame. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Every SSE frame's `data:` lines, `\r\n` normalized and the trailing partial frame flushed at EOF. */
async function* readSSEFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<string[]> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffered = '';
  const dataLinesOf = (frame: string): string[] =>
    frame
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim());
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let cut = buffered.indexOf('\n\n');
      while (cut !== -1) {
        const frame = buffered.slice(0, cut);
        buffered = buffered.slice(cut + 2);
        const lines = dataLinesOf(frame);
        if (lines.length > 0) yield lines;
        cut = buffered.indexOf('\n\n');
      }
    }
    buffered += decoder.decode();
    const tail = dataLinesOf(buffered);
    if (tail.length > 0) yield tail;
  } finally {
    reader.releaseLock();
  }
}

/** An OpenAI Chat Completions SSE stream as IrEvents. */
async function* readOpenAIStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<IrEvent> {
  const toolSlotOf = new Map<number, number>();
  let nextSlot = 0;
  let outputTokens = 0;
  let pendingStop: IrStop | null = null;
  for await (const lines of readSSEFrames(stream)) {
    for (const payload of lines) {
      if (payload === '[DONE]') {
        yield { kind: 'finished', stop: pendingStop ?? 'done', outputTokens };
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        yield { kind: 'input_tokens', count: asPositiveInt(chunk.usage.prompt_tokens) };
        outputTokens = asPositiveInt(chunk.usage.completion_tokens);
      }
      const choice = Array.isArray(chunk.choices) && isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string' && delta.content !== '') yield { kind: 'text_delta', text: delta.content };
      for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolSlotOf.size;
        let slot = toolSlotOf.get(upstreamIndex);
        if (slot === undefined) {
          slot = nextSlot;
          nextSlot += 1;
          toolSlotOf.set(upstreamIndex, slot);
          const fn = isRecord(call.function) ? call.function : {};
          yield { kind: 'tool_call_start', slot, callId: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${slot}`, toolName: typeof fn.name === 'string' ? fn.name : '' };
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') yield { kind: 'tool_call_delta', slot, argsJsonFragment: fn.arguments };
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopOf(choice.finish_reason);
    }
  }
}

/** An Anthropic Messages SSE stream as IrEvents. */
async function* readAnthropicStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<IrEvent> {
  const toolSlotOf = new Map<number, number>();
  let nextSlot = 0;
  let outputTokens = 0;
  let pendingStop: IrStop | null = null;
  for await (const lines of readSSEFrames(stream)) {
    for (const payload of lines) {
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (chunk.type === 'message_start' && isRecord(chunk.message) && isRecord(chunk.message.usage)) {
        yield { kind: 'input_tokens', count: asPositiveInt(chunk.message.usage.input_tokens) };
      } else if (chunk.type === 'content_block_start' && typeof chunk.index === 'number' && isRecord(chunk.content_block) && chunk.content_block.type === 'tool_use') {
        const slot = nextSlot;
        nextSlot += 1;
        toolSlotOf.set(chunk.index, slot);
        yield { kind: 'tool_call_start', slot, callId: typeof chunk.content_block.id === 'string' ? chunk.content_block.id : `toolu_${slot}`, toolName: typeof chunk.content_block.name === 'string' ? chunk.content_block.name : '' };
      } else if (chunk.type === 'content_block_delta' && isRecord(chunk.delta)) {
        if (chunk.delta.type === 'text_delta' && typeof chunk.delta.text === 'string') yield { kind: 'text_delta', text: chunk.delta.text };
        else if (chunk.delta.type === 'input_json_delta' && typeof chunk.delta.partial_json === 'string' && typeof chunk.index === 'number') {
          const slot = toolSlotOf.get(chunk.index);
          if (slot !== undefined) yield { kind: 'tool_call_delta', slot, argsJsonFragment: chunk.delta.partial_json };
        }
      } else if (chunk.type === 'message_delta') {
        if (isRecord(chunk.delta) && typeof chunk.delta.stop_reason === 'string') pendingStop = stopOf(chunk.delta.stop_reason);
        if (isRecord(chunk.usage)) outputTokens = asPositiveInt(chunk.usage.output_tokens);
      } else if (chunk.type === 'message_stop') {
        yield { kind: 'finished', stop: pendingStop ?? 'done', outputTokens };
      }
    }
  }
}

/** A Gemini `:streamGenerateContent` SSE stream as IrEvents (each frame holds one candidate set). */
async function* readGeminiStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<IrEvent> {
  let outputTokens = 0;
  let pendingStop: IrStop | null = null;
  let announcedTool = false;
  for await (const lines of readSSEFrames(stream)) {
    for (const payload of lines) {
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usageMetadata)) {
        yield { kind: 'input_tokens', count: asPositiveInt(chunk.usageMetadata.promptTokenCount) };
        outputTokens = asPositiveInt(chunk.usageMetadata.candidatesTokenCount);
      }
      const candidate = Array.isArray(chunk.candidates) && isRecord(chunk.candidates[0]) ? chunk.candidates[0] : null;
      if (candidate === null) continue;
      const content = isRecord(candidate.content) ? candidate.content : {};
      for (const part of Array.isArray(content.parts) ? content.parts : []) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string' && part.text !== '') {
          yield { kind: 'text_delta', text: part.text };
        } else if (isRecord(part.functionCall) && typeof part.functionCall.name === 'string') {
          if (!announcedTool) {
            announcedTool = true;
            yield { kind: 'tool_call_start', slot: 0, callId: `toolu_${part.functionCall.name}`, toolName: part.functionCall.name };
          }
          yield { kind: 'tool_call_delta', slot: 0, argsJsonFragment: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) };
        }
      }
      if (typeof candidate.finishReason === 'string') pendingStop = stopOf(candidate.finishReason);
    }
  }
  yield { kind: 'finished', stop: pendingStop ?? 'done', outputTokens };
}

/** A provider's own response stream as IrEvents, dispatched by that provider's wire. */
export function canonicalEventsFromProviderStream(wire: 'openai' | 'anthropic' | 'gemini', stream: ReadableStream<Uint8Array>): AsyncGenerator<IrEvent> {
  if (wire === 'openai') return readOpenAIStream(stream);
  if (wire === 'anthropic') return readAnthropicStream(stream);
  return readGeminiStream(stream);
}

// ---------------------------------------------------------------------------------------------------------
// IrEvent stream -> client wire's own SSE shape.
// ---------------------------------------------------------------------------------------------------------

/** IrEvents as an OpenAI Chat Completions SSE stream, naming `model` as the client itself asked for it. */
export async function* openAISSEFromCanonical(events: AsyncIterable<IrEvent>, model: string): AsyncGenerator<string> {
  const chunkOf = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_LIST_EPOCH_SECONDS, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  yield sseEvent(null, chunkOf({ role: 'assistant', content: '' }, null));
  const slotToOpenAIIndex = new Map<number, number>();
  let inputTokens = 0;
  for await (const event of events) {
    if (event.kind === 'input_tokens') {
      inputTokens = event.count; // held for the final usage chunk; OpenAI's own first chunk carries none
      continue;
    }
    if (event.kind === 'text_delta') {
      yield sseEvent(null, chunkOf({ content: event.text }, null));
    } else if (event.kind === 'tool_call_start') {
      const index = slotToOpenAIIndex.size;
      slotToOpenAIIndex.set(event.slot, index);
      yield sseEvent(null, chunkOf({ tool_calls: [{ index, id: event.callId, type: 'function', function: { name: event.toolName, arguments: '' } }] }, null));
    } else if (event.kind === 'tool_call_delta') {
      const index = slotToOpenAIIndex.get(event.slot) ?? event.slot;
      yield sseEvent(null, chunkOf({ tool_calls: [{ index, function: { arguments: event.argsJsonFragment } }] }, null));
    } else {
      yield sseEvent(null, chunkOf({}, OPENAI_FINISH_REASON[event.stop]));
      if (event.outputTokens !== null) {
        yield sseEvent(null, {
          id: 'chatcmpl-aggregate',
          object: 'chat.completion.chunk',
          created: FIXED_LIST_EPOCH_SECONDS,
          model,
          choices: [],
          usage: { prompt_tokens: inputTokens, completion_tokens: event.outputTokens, total_tokens: inputTokens + event.outputTokens },
        });
      }
      yield 'data: [DONE]\n\n';
    }
  }
}

/**
 * IrEvents as the Anthropic event sequence, naming `model` as the client itself asked for it.
 * `message_start` is held back until either an `input_tokens` event or the first real content event
 * arrives, so its `usage.input_tokens` can carry the real count when the provider wire gave it early (an
 * anthropic-wire provider's own `message_start` does; an openai- or gemini-wire one never knows its input
 * tokens before the answer is done, same as the real Anthropic API reached through either, so it stays 0
 * there -- nothing later can correct it, because Anthropic's own `message_delta` carries no input count).
 */
export async function* anthropicSSEFromCanonical(events: AsyncIterable<IrEvent>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let started = false;
  let textSlot = -1;
  let nextSlot = 0;
  const toolSlotOf = new Map<number, number>();

  const messageStart = (): string =>
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    });

  for await (const event of events) {
    if (event.kind === 'input_tokens') {
      inputTokens = event.count;
      continue;
    }
    if (!started) {
      started = true;
      yield messageStart();
    }
    if (event.kind === 'text_delta') {
      if (textSlot === -1) {
        textSlot = nextSlot;
        nextSlot += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textSlot, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textSlot, delta: { type: 'text_delta', text: event.text } });
    } else if (event.kind === 'tool_call_start') {
      if (textSlot !== -1) {
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textSlot });
        textSlot = -1;
      }
      const index = nextSlot;
      nextSlot += 1;
      toolSlotOf.set(event.slot, index);
      yield sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: event.callId, name: event.toolName, input: {} } });
    } else if (event.kind === 'tool_call_delta') {
      const index = toolSlotOf.get(event.slot);
      if (index !== undefined) yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: event.argsJsonFragment } });
    } else if (event.kind === 'finished') {
      if (textSlot !== -1) {
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textSlot });
        textSlot = -1;
      }
      for (const index of [...toolSlotOf.values()].sort((a, b) => a - b)) yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
      yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: ANTHROPIC_STOP_REASON[event.stop], stop_sequence: null }, usage: { output_tokens: event.outputTokens ?? 0 } });
      yield sseEvent('message_stop', { type: 'message_stop' });
    }
  }
  if (!started) yield messageStart(); // guard for an empty stream; every real provider answer sends 'finished'
}
