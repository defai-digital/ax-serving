import assert from "node:assert/strict";
import test from "node:test";

import { AxServingClient } from "../dist/index.js";

test("chatCompletionsCreate preserves assistant tool-call history", async () => {
  let capturedBody = "";
  const client = new AxServingClient({
    baseURL: "http://127.0.0.1:18080/",
    apiKey: "secret",
    fetchImpl: async (url, init) => {
      assert.equal(url, "http://127.0.0.1:18080/v1/chat/completions");
      assert.equal(init.headers.get("authorization"), "Bearer secret");
      capturedBody = init.body;
      return new Response(
        JSON.stringify({
          id: "chatcmpl-test",
          object: "chat.completion",
          created: 0,
          model: "default",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "ok" },
              finish_reason: "stop",
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    },
  });

  await client.chatCompletionsCreate({
    model: "default",
    messages: [
      { role: "user", content: "call the tool" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "lookup", arguments: "{}" },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "result" },
    ],
  });

  const body = JSON.parse(capturedBody);
  assert.equal(body.stream, false);
  assert.equal(body.messages[1].content, null);
  assert.equal(body.messages[1].tool_calls[0].id, "call_1");
  assert.equal(body.messages[2].tool_call_id, "call_1");
});
