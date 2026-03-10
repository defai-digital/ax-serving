# @defai-digital/ax-serving

TypeScript SDK for AX Serving using OpenAI-compatible REST endpoints.

## Why Zod

This SDK validates request and response payloads with Zod to match the rest of the JS stack and catch integration bugs early.

## Install

Published package:

```bash
npm install @defai-digital/ax-serving
```

Local workspace package:

```bash
npm install ./sdk/javascript
```

## Usage

```ts
import { AxServingClient } from "@defai-digital/ax-serving";

const client = new AxServingClient({
  baseURL: "http://127.0.0.1:18080",
  apiKey: process.env.AXS_API_KEY,
});

const resp = await client.chatCompletionsCreate({
  model: "default",
  messages: [{ role: "user", content: "Say hello in one sentence." }],
  max_tokens: 32,
  repeat_penalty: 1.1,
});

console.log(resp.choices[0]?.message?.content ?? "");
```

Streaming:

```ts
for await (const chunk of client.chatCompletionsStream({
  model: "default",
  messages: [{ role: "user", content: "Count from 1 to 5." }],
  max_tokens: 64,
})) {
  const text = chunk.choices[0]?.delta?.content ?? "";
  process.stdout.write(text);
}
```

List models:

```ts
const models = await client.modelsList();
console.log(models.data.map((m) => m.id));
```
