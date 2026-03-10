import { z } from "zod";

export const ChatMessageSchema = z.object({
  role: z.string(),
  content: z.union([z.string(), z.array(z.unknown())]),
});

export const ChatCompletionRequestSchema = z.object({
  model: z.string().min(1),
  messages: z.array(ChatMessageSchema).min(1),
  stream: z.boolean().optional(),
  temperature: z.number().min(0).max(2).optional(),
  max_tokens: z.number().int().positive().optional(),
  top_p: z.number().gt(0).lte(1).optional(),
  top_k: z.number().int().positive().optional(),
  seed: z.number().int().nonnegative().optional(),
  repeat_penalty: z.number().gt(0).lte(10).optional(),
  stop: z.union([z.string(), z.array(z.string())]).optional(),
  frequency_penalty: z.number().min(-2).max(2).optional(),
  presence_penalty: z.number().min(-2).max(2).optional(),
  grammar: z.string().optional(),
  response_format: z
    .object({
      type: z.enum(["text", "json_object"]),
    })
    .optional(),
  mirostat: z.union([z.literal(0), z.literal(1), z.literal(2)]).optional(),
  mirostat_tau: z.number().optional(),
  mirostat_eta: z.number().optional(),
  tools: z.array(z.unknown()).optional(),
  tool_choice: z.unknown().optional(),
  cache: z.enum(["enable", "disable"]).optional(),
  cache_ttl: z.string().optional(),
  logprobs: z.boolean().optional(),
  top_logprobs: z.number().int().min(0).max(20).optional(),
});

export const ChatCompletionChoiceSchema = z.object({
  index: z.number().int().nonnegative(),
  message: z
    .object({
      role: z.string(),
      content: z.string().nullable().optional(),
      tool_calls: z.array(z.unknown()).optional(),
    })
    .nullable()
    .optional(),
  delta: z
    .object({
      role: z.string().optional(),
      content: z.string().nullable().optional(),
      tool_calls: z.array(z.unknown()).optional(),
    })
    .nullable()
    .optional(),
  finish_reason: z.string().nullable().optional(),
  logprobs: z.unknown().optional(),
});

export const ChatCompletionResponseSchema = z.object({
  id: z.string(),
  object: z.string(),
  created: z.number(),
  model: z.string(),
  choices: z.array(ChatCompletionChoiceSchema),
  usage: z
    .object({
      prompt_tokens: z.number().int().nonnegative(),
      completion_tokens: z.number().int().nonnegative(),
      total_tokens: z.number().int().nonnegative(),
    })
    .optional(),
});

export const ModelListResponseSchema = z.object({
  object: z.literal("list"),
  data: z.array(
    z.object({
      id: z.string(),
      object: z.string(),
      created: z.number(),
      owned_by: z.string(),
    }),
  ),
});

export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type ChatCompletionRequest = z.infer<typeof ChatCompletionRequestSchema>;
export type ChatCompletionChoice = z.infer<typeof ChatCompletionChoiceSchema>;
export type ChatCompletionResponse = z.infer<typeof ChatCompletionResponseSchema>;
export type ModelListResponse = z.infer<typeof ModelListResponseSchema>;
