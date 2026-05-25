# AX Code Integration

Status: Proposed integration contract

AX Code should integrate with AX Serving through stable, machine-readable CLI and HTTP
contracts. AX Serving remains a headless Rust runtime; AX Code owns any interactive agent,
TUI, or support-console experience.

## Recommended Boundary

Use AX Serving for:

- Model serving and orchestration.
- Host diagnostics.
- Hardware-based tuning recommendations.
- Health, metrics, and model lifecycle APIs.

Use AX Code for:

- Guided setup and repair workflows.
- Agent explanations of diagnostic failures.
- Support-console UI, including any OpenTUI surface.
- Support bundle generation and ticket summaries.
- Repository-aware changes to config, docs, deployment scripts, and tests.

This keeps the serving runtime scriptable and testable while letting AX Code provide a
higher-level operator experience.

## CLI Contracts

The first supported automation contracts are:

```bash
ax-serving doctor --json
ax-serving tune --dry-run --json
ax-serving tune --output serving.toml --json
ax-serving config validate --json
ax-serving status --url http://127.0.0.1:18080 --json
ax-serving smoke-test --url http://127.0.0.1:18080 --model default --json
```

`doctor --json` returns a redacted diagnostic report with check status, details, and
remediation hints. It exits nonzero when a failed check means serving would not start
safely.

`tune --json` returns the detected hardware profile, recommended scheduler settings, the
rendered TOML, and the written output path when a file is created.

`config validate --json` validates the resolved serving config and returns a compact
runtime summary that is safe for setup tools to display.

`status --json` probes `/health`, `/v1/models`, and `/v1/metrics` on a running worker or
gateway. Use `--api-key` when authentication is enabled; otherwise it uses the first
token in `AXS_API_KEY` when available.

`smoke-test --json` sends a minimal non-streaming `/v1/chat/completions` request and
returns the HTTP status, latency, and parsed JSON response.

## AX Code SDK Pattern

For a TypeScript integration, define AX Serving as explicit AX Code SDK tools instead of
embedding serving logic in prompts.

```ts
import { createAgent, tool } from "@ax-code/sdk"
import { z } from "zod"
import { execa } from "execa"

const axServingDoctor = tool({
  name: "ax_serving_doctor",
  description: "Run AX Serving diagnostics and return the structured report.",
  parameters: z.object({}),
  execute: async () => {
    const result = await execa("ax-serving", ["doctor", "--json"], {
      reject: false,
    })
    return {
      exitCode: result.exitCode,
      stdout: JSON.parse(result.stdout),
      stderr: result.stderr,
    }
  },
})

const axServingTune = tool({
  name: "ax_serving_tune",
  description: "Generate an AX Serving hardware-based tuning recommendation.",
  parameters: z.object({
    dryRun: z.boolean().default(true),
    output: z.string().optional(),
  }),
  execute: async ({ dryRun, output }) => {
    const args = ["tune", "--json"]
    if (dryRun) args.push("--dry-run")
    if (output) args.push("--output", output)

    const result = await execa("ax-serving", args, { reject: false })
    return {
      exitCode: result.exitCode,
      stdout: JSON.parse(result.stdout),
      stderr: result.stderr,
    }
  },
})

const agent = await createAgent({
  directory: ".",
  tools: [axServingDoctor, axServingTune],
})
```

The AX Code prompt should ask the agent to call these tools, inspect the structured
results, and explain only the relevant remediation steps.

## OpenTUI Usage

OpenTUI is appropriate for an optional AX Code support console. It should call the same
tools above and display:

- Environment readiness checklist.
- Hardware profile and generated tuning values.
- Model path validation.
- Worker and gateway start commands.
- Smoke-test result.
- Redacted diagnostics for support.

Do not add OpenTUI, Bun, or TypeScript dependencies to the AX Serving Rust workspace.

## Next Contracts

Useful follow-up contracts for AX Code automation:

- `ax-serving support-bundle --redact --output <path>`
