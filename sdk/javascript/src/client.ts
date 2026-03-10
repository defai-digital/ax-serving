import { z } from "zod";

import {
  ChatCompletionRequestSchema,
  ChatCompletionResponse,
  ChatCompletionResponseSchema,
  ModelListResponse,
  ModelListResponseSchema,
} from "./types.js";

export type AxClientOptions = {
  baseURL?: string;
  apiKey?: string;
  fetchImpl?: typeof fetch;
};

export class AxServingClient {
  private readonly baseURL: string;
  private readonly apiKey?: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: AxClientOptions = {}) {
    this.baseURL = (options.baseURL ?? "http://127.0.0.1:18080").replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async modelsList(): Promise<ModelListResponse> {
    const data = await this.requestJson("/v1/models", {
      method: "GET",
    });
    return ModelListResponseSchema.parse(data);
  }

  async chatCompletionsCreate(
    input: unknown,
  ): Promise<ChatCompletionResponse> {
    const req = ChatCompletionRequestSchema.parse(input);
    const data = await this.requestJson("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({ ...req, stream: false }),
    });
    return ChatCompletionResponseSchema.parse(data);
  }

  async *chatCompletionsStream(
    input: unknown,
  ): AsyncGenerator<ChatCompletionResponse, void, unknown> {
    const req = ChatCompletionRequestSchema.parse(input);
    const response = await this.fetchImpl(`${this.baseURL}/v1/chat/completions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ ...req, stream: true }),
    });
    if (!response.ok) {
      throw await this.buildHttpError(response);
    }
    if (!response.body) {
      throw new Error("stream response has no body");
    }

    for await (const chunk of parseSseData(response.body)) {
      if (chunk === "[DONE]") {
        return;
      }
      const parsed = parseJsonSafe(chunk);
      yield ChatCompletionResponseSchema.parse(parsed);
    }
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private async requestJson(path: string, init: RequestInit): Promise<unknown> {
    const response = await this.fetchImpl(`${this.baseURL}${path}`, {
      ...init,
      headers: mergeHeaders(this.headers(), init.headers),
    });
    if (!response.ok) {
      throw await this.buildHttpError(response);
    }
    return response.json();
  }

  private async buildHttpError(response: Response): Promise<Error> {
    const body = await response.text();
    return new Error(`HTTP ${response.status}: ${body}`);
  }
}

function mergeHeaders(
  base: Record<string, string>,
  extra?: HeadersInit,
): Headers {
  const headers = new Headers(base);
  if (!extra) {
    return headers;
  }
  const additional = new Headers(extra);
  additional.forEach((value, key) => {
    headers.set(key, value);
  });
  return headers;
}

async function* parseSseData(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<string, void, unknown> {
  const decoder = new TextDecoder();
  const reader = stream.getReader();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      if (buffer.length > 0) {
        const tail = collectSsePayloads(buffer, true);
        for (const payload of tail.payloads) {
          yield payload;
        }
      }
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const { remainder, payloads } = collectSsePayloads(buffer, false);
    buffer = remainder;
    for (const payload of payloads) {
      yield payload;
    }
  }
}

function collectSsePayloads(
  raw: string,
  flushTail: boolean,
): { remainder: string; payloads: string[] } {
  const normalized = raw.replace(/\r\n/g, "\n");
  const segments = normalized.split("\n\n");
  const payloads: string[] = [];

  const completeCount = flushTail ? segments.length : Math.max(segments.length - 1, 0);
  for (let i = 0; i < completeCount; i += 1) {
    const event = segments[i];
    const lines: string[] = [];
    for (const line of event.split("\n")) {
      if (!line.startsWith("data:")) {
        continue;
      }
      lines.push(line.slice(5).trimStart());
    }
    const payload = lines.join("\n").trim();
    if (payload.length > 0) {
      payloads.push(payload);
    }
  }

  return {
    remainder: flushTail ? "" : (segments.at(-1) ?? ""),
    payloads,
  };
}

function parseJsonSafe(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(
      `failed to parse SSE JSON chunk: ${(error as Error).message}; chunk=${raw}`,
    );
  }
}

export const schema = {
  chatCompletionsRequest: ChatCompletionRequestSchema,
  chatCompletionsResponse: ChatCompletionResponseSchema,
  modelsListResponse: ModelListResponseSchema,
} as const;

export function validateWith<T>(schemaDef: z.ZodSchema<T>, value: unknown): T {
  return schemaDef.parse(value);
}
