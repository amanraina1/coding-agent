import { describe, expect, test } from "bun:test";
import type { OpenRouter } from "@openrouter/sdk";
import { runAgentLoop } from "./agent.ts";
import { executeToolCall, getToolsForDepth } from "./tools.ts";

describe("subagent orchestration", () => {
  test("getToolsForDepth exposes spawn_subagent only below maxDepth", () => {
    const rootTools = getToolsForDepth(0, 1).map((tool) => tool.name);
    const subagentTools = getToolsForDepth(1, 1).map((tool) => tool.name);

    expect(rootTools).toEqual(["get_weather", "execute_bash", "spawn_subagent"]);
    expect(subagentTools).toEqual(["get_weather", "execute_bash"]);
  });

  test("executeToolCall blocks spawn_subagent when runner is unavailable", async () => {
    const output = await executeToolCall({
      type: "function_call",
      id: "fc_1",
      callId: "call_1",
      name: "spawn_subagent",
      arguments: JSON.stringify({ task: "Do work" }),
    });

    expect(output.type).toBe("function_call_output");
    expect(output.callId).toBe("call_1");
    expect(typeof output.output).toBe("string");
    expect(JSON.parse(String(output.output))).toEqual({
      status: "error",
      error: "spawn_subagent is not permitted at this agent depth.",
    });
  });

  test("executeToolCall returns error payload on invalid arguments", async () => {
    const output = await executeToolCall(
      {
        type: "function_call",
        id: "fc_2",
        callId: "call_2",
        name: "spawn_subagent",
        arguments: JSON.stringify({ task: "" }),
      },
      async () => ({
        status: "completed",
        role: "Worker",
        iterations: 1,
        result: "unused",
      }),
    );

    expect(typeof output.output).toBe("string");
    expect(JSON.parse(String(output.output))).toEqual({
      error: 'spawn_subagent requires a non-empty string "task"',
    });
  });

  test("orchestrator spawns parallel subagents in non-streaming mode", async () => {
    const recordedCalls: Array<{ toolNames: string[]; userPrompt: string }> = [];

    const mockClient = {
      responses: {
        send: async ({ responsesRequest }: { responsesRequest: { tools: Array<{ name: string }>; input: Array<{ role?: string; content?: unknown }> } }) => {
          const toolNames = responsesRequest.tools.map((t) => t.name);
          const userMessage = responsesRequest.input.find((m) => m.role === "user");
          const userPrompt = typeof userMessage?.content === "string" ? userMessage.content : "";
          recordedCalls.push({ toolNames, userPrompt });

          const hasToolOutput = responsesRequest.input.some(
            (item) => (item as { type?: string }).type === "function_call_output",
          );

          // Orchestrator turn 1: spawn 2 parallel subagents
          if (toolNames.includes("spawn_subagent") && !hasToolOutput) {
            return {
              status: "completed",
              output: [
                {
                  type: "function_call",
                  id: "fc_sub_1",
                  callId: "call_sub_1",
                  name: "spawn_subagent",
                  arguments: JSON.stringify({
                    role: "Repo Inspector",
                    task: "Run pwd via bash",
                    context: "Check current directory",
                  }),
                },
                {
                  type: "function_call",
                  id: "fc_sub_2",
                  callId: "call_sub_2",
                  name: "spawn_subagent",
                  arguments: JSON.stringify({
                    role: "Echo Worker",
                    task: "Echo hello via bash",
                  }),
                },
              ],
            };
          }

          // Subagent turn 1: run execute_bash
          if (!toolNames.includes("spawn_subagent") && !hasToolOutput) {
            const cmd = userPrompt.includes("pwd") ? "pwd" : "echo hello";
            return {
              status: "completed",
              output: [
                {
                  type: "function_call",
                  id: "fc_bash",
                  callId: "call_bash",
                  name: "execute_bash",
                  arguments: JSON.stringify({ command: cmd }),
                },
              ],
            };
          }

          // Subagent turn 2: return final answer
          if (!toolNames.includes("spawn_subagent") && hasToolOutput) {
            const toolOutputItem = responsesRequest.input.find(
              (item) => (item as { type?: string }).type === "function_call_output",
            ) as { output: string };
            return {
              status: "completed",
              output: [
                {
                  type: "message",
                  id: "msg_sub",
                  role: "assistant",
                  content: [{ type: "output_text", text: `Subagent output: ${toolOutputItem.output}`, annotations: [] }],
                },
              ],
            };
          }

          // Orchestrator turn 2: summarize subagent results
          const subOutputs = responsesRequest.input
            .filter((item) => (item as { type?: string }).type === "function_call_output")
            .map((item) => JSON.parse((item as { output: string }).output) as { role: string; result: string });

          return {
            status: "completed",
            output: [
              {
                type: "message",
                id: "msg_root",
                role: "assistant",
                content: [
                  {
                    type: "output_text",
                    text: subOutputs.map((s) => `[${s.role}] ${s.result}`).join(" | "),
                    annotations: [],
                  },
                ],
              },
            ],
          };
        },
      },
    } as unknown as OpenRouter;

    const result = await runAgentLoop({
      client: mockClient,
      systemPrompt: "Orchestrator",
      userPrompt: "Run two subtasks",
      stream: false,
    });

    expect(result.status).toBe("completed");
    expect(result.iterations).toBe(2);
    expect(result.text).toContain("[Repo Inspector]");
    expect(result.text).toContain("coding-agent");
    expect(result.text).toContain("[Echo Worker]");
    expect(result.text).toContain("hello");

    // Verify subagents never received spawn_subagent
    const subagentCalls = recordedCalls.filter((c) => !c.toolNames.includes("spawn_subagent"));
    expect(subagentCalls.length).toBe(4); // 2 turns x 2 subagents
  });

  test("orchestrator spawns subagent in streaming mode", async () => {
    const streamedDeltas: string[] = [];

    const mockClient = {
      responses: {
        send: async ({ responsesRequest }: { responsesRequest: { tools: Array<{ name: string }>; input: Array<{ type?: string }> } }) => {
          const isRoot = responsesRequest.tools.some((t) => t.name === "spawn_subagent");
          const hasToolOutput = responsesRequest.input.some((item) => item.type === "function_call_output");

          async function* streamEvents(events: unknown[]) {
            for (const event of events) {
              yield event;
            }
          }

          if (isRoot && !hasToolOutput) {
            const fcItem = {
              type: "function_call",
              id: "fc_stream_sub",
              callId: "call_stream_sub",
              name: "spawn_subagent",
              arguments: JSON.stringify({ role: "Streamer", task: "Say done" }),
            };
            return streamEvents([
              { type: "response.output_item.done", item: fcItem },
              { type: "response.completed", response: { output: [fcItem] } },
            ]);
          }

          if (!isRoot) {
            const msgItem = {
              type: "message",
              id: "msg_stream_sub",
              role: "assistant",
              content: [{ type: "output_text", text: "subagent streamed result", annotations: [] }],
            };
            return streamEvents([
              { type: "response.output_text.delta", delta: "subagent streamed result" },
              { type: "response.output_item.done", item: msgItem },
              { type: "response.completed", response: { output: [msgItem] } },
            ]);
          }

          const rootMsg = {
            type: "message",
            id: "msg_stream_root",
            role: "assistant",
            content: [{ type: "output_text", text: "Root final answer", annotations: [] }],
          };
          return streamEvents([
            { type: "response.output_text.delta", delta: "Root " },
            { type: "response.output_text.delta", delta: "final answer" },
            { type: "response.output_item.done", item: rootMsg },
            { type: "response.completed", response: { output: [rootMsg] } },
          ]);
        },
      },
    } as unknown as OpenRouter;

    const result = await runAgentLoop({
      client: mockClient,
      systemPrompt: "Orchestrator",
      userPrompt: "Run streamed subtask",
      stream: true,
      onTextDelta: (delta) => streamedDeltas.push(delta),
    });

    expect(result.status).toBe("completed");
    expect(result.text).toBe("Root final answer");
    // Only root deltas should stream to onTextDelta (subagent output must not interleave)
    expect(streamedDeltas.join("")).toBe("Root final answer");
  });
});
