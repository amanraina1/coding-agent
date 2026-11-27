import { OpenRouter } from "@openrouter/sdk";
import type {
  InputsMessage,
  InputsUnion1,
  OutputFunctionCallItem,
  OutputItems,
  OutputMessageItem,
  ResponsesRequestToolFunction,
} from "@openrouter/sdk/models";
import {
  executeToolCall,
  getToolsForDepth,
  type SubagentRequest,
  type SubagentResult,
  type SubagentRunner,
} from "./tools.ts";

export interface CallLLMOptions {
  model: string;
  stream: boolean;
  onTextDelta?: (delta: string) => void;
  onToolCall?: (toolCall: OutputFunctionCallItem) => void;
}

export interface RunAgentOptions {
  client?: OpenRouter;
  model?: string;
  systemPrompt: string;
  userPrompt: string;
  stream?: boolean;
  depth?: number;
  maxDepth?: number;
  maxIterations?: number;
  onTextDelta?: (delta: string) => void;
  onToolCall?: (toolCall: OutputFunctionCallItem) => void;
}

export interface AgentRunResult {
  status: "completed" | "max_iterations";
  iterations: number;
  text: string;
}

export const DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT =
  "You are a helpful orchestrator assistant with access to tools: get_weather, execute_bash, and spawn_subagent. " +
  "REMEMBER THE EXACT TOOL NAMES AND NEVER INVENT OTHER NAMES. " +
  "For large or multi-part tasks, break the work into focused subtasks and delegate them using spawn_subagent " +
  "(multiple spawn_subagent calls in one turn run in parallel). Combine the subagents' results into a clear final response.";

export const DEFAULT_ORCHESTRATOR_USER_PROMPT =
  "Inspect this project directory and check today's weather in Gurgaon by delegating those two independent tasks to separate subagents in parallel, then summarize both results.";

const DEFAULT_MODEL = "openrouter/free";
const DEFAULT_MAX_DEPTH = 1;
const DEFAULT_ROOT_MAX_ITERATIONS = 20;
const DEFAULT_SUBAGENT_MAX_ITERATIONS = 10;

let nextSubagentId = 1;

export function createOpenRouterClient(): OpenRouter {
  return new OpenRouter({
    apiKey: process.env.API_KEY,
  });
}

function toInputItems(outputItems: OutputItems[]): InputsUnion1[] {
  const result: InputsUnion1[] = [];

  for (const item of outputItems) {
    if (item.type === "UNKNOWN") {
      continue;
    }

    if (item.type === "message") {
      const normalizedContent = item.content.filter(
        (part): part is Extract<(typeof item.content)[number], { type: "output_text" | "refusal" }> =>
          part.type === "output_text" || part.type === "refusal",
      );
      const messageItem: InputsMessage = {
        ...item,
        content: normalizedContent,
      };
      result.push(messageItem);
      continue;
    }

    result.push(item as InputsUnion1);
  }

  return result;
}

function extractFinalText(output: OutputItems[]): string {
  const message = [...output]
    .reverse()
    .find((item): item is OutputMessageItem => item.type === "message");

  if (!message) {
    return "";
  }

  return message.content
    .filter(
      (part): part is Extract<(typeof message.content)[number], { type: "output_text" }> =>
        part.type === "output_text",
    )
    .map((part) => part.text)
    .join("");
}

/**
 * Sends the conversation history to OpenRouter in streaming or non-streaming mode,
 * appends the resulting output items to history, and returns them.
 */
export async function callLLM(
  client: OpenRouter,
  conversationHistory: InputsUnion1[],
  tools: ResponsesRequestToolFunction[],
  options: CallLLMOptions,
): Promise<OutputItems[]> {
  const result = await client.responses.send({
    responsesRequest: {
      stream: options.stream,
      tools,
      model: options.model,
      input: conversationHistory,
    },
  });

  if ("output" in result) {
    if (result.status !== "completed") {
      throw new Error(result.error?.message ?? `Response ended as ${result.status}`);
    }

    const outputItems = result.output;
    for (const item of outputItems) {
      if (item.type === "function_call") {
        options.onToolCall?.(item);
      }
    }

    conversationHistory.push(...toInputItems(outputItems));
    return outputItems;
  }

  let completedOutput: OutputItems[] | null = null;
  const fallbackOutputItems: OutputItems[] = [];

  for await (const event of result) {
    switch (event.type) {
      case "response.failed":
        throw new Error(event.response.error?.message ?? "Response failed");

      case "response.incomplete":
        throw new Error(
          `Response incomplete: ${event.response.incompleteDetails?.reason ?? "unknown reason"}`,
        );

      case "response.output_text.delta":
        if (event.delta) {
          options.onTextDelta?.(event.delta);
        }
        break;

      case "response.output_item.done": {
        const item = event.item as OutputItems | undefined;
        if (item) {
          fallbackOutputItems.push(item);
          if (item.type === "function_call") {
            options.onToolCall?.(item);
          }
        }
        break;
      }

      case "response.completed":
        completedOutput = event.response.output;
        break;
    }
  }

  if (completedOutput === null) {
    throw new Error("Stream ended before receiving response.completed event");
  }

  const outputItems = completedOutput.length > 0 ? completedOutput : fallbackOutputItems;
  if (outputItems.length === 0) {
    throw new Error("Stream ended without producing any output items");
  }

  conversationHistory.push(...toInputItems(outputItems));
  return outputItems;
}

function buildSubagentSystemPrompt(role: string): string {
  return (
    `You are a focused subagent operating as "${role}". ` +
    "Complete the assigned task using your available tools (get_weather, execute_bash). " +
    "DO NOT INVENT TOOL NAMES OR CLAIM RESULTS YOU DID NOT OBSERVE. " +
    "Return a clear, concise final summary of your findings or completed work."
  );
}

function buildSubagentUserPrompt(request: SubagentRequest): string {
  if (!request.context) {
    return request.task;
  }
  return `Context:\n${request.context}\n\nTask:\n${request.task}`;
}

function createSubagentRunner(parent: {
  client: OpenRouter;
  model: string;
  stream: boolean;
  depth: number;
  maxDepth: number;
}): SubagentRunner {
  return async (request: SubagentRequest): Promise<SubagentResult> => {
    const id = nextSubagentId++;
    const role = request.role ?? "Subagent";

    console.log(`\n[Subagent #${id} | ${role}] Started: ${request.task}`);

    try {
      const subagentRun = await runAgentLoop({
        client: parent.client,
        model: parent.model,
        systemPrompt: buildSubagentSystemPrompt(role),
        userPrompt: buildSubagentUserPrompt(request),
        stream: parent.stream,
        depth: parent.depth + 1,
        maxDepth: parent.maxDepth,
        maxIterations: DEFAULT_SUBAGENT_MAX_ITERATIONS,
        onToolCall: (toolCall) => {
          console.log(`[Subagent #${id}] Calling tool: ${toolCall.name}(${toolCall.arguments})`);
        },
      });

      console.log(
        `[Subagent #${id} | ${role}] ${subagentRun.status} in ${subagentRun.iterations} turn(s)`,
      );

      return {
        status: subagentRun.status,
        role,
        iterations: subagentRun.iterations,
        result: subagentRun.text,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[Subagent #${id} | ${role}] Failed: ${message}`);
      return {
        status: "error",
        role,
        iterations: 0,
        result: message,
      };
    }
  };
}

/**
 * Runs an agent loop until the model produces a final message with no tool calls
 * or reaches maxIterations. Root agents can spawn isolated subagents up to maxDepth.
 */
export async function runAgentLoop(options: RunAgentOptions): Promise<AgentRunResult> {
  const client = options.client ?? createOpenRouterClient();
  const model = options.model ?? DEFAULT_MODEL;
  const stream = options.stream ?? true;
  const depth = options.depth ?? 0;
  const maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH;
  const maxIterations = options.maxIterations ?? DEFAULT_ROOT_MAX_ITERATIONS;

  const availableTools = getToolsForDepth(depth, maxDepth);
  const runSubagent =
    depth < maxDepth
      ? createSubagentRunner({ client, model, stream, depth, maxDepth })
      : undefined;

  const conversationHistory: InputsUnion1[] = [
    {
      role: "system",
      content: options.systemPrompt,
    },
    {
      role: "user",
      content: options.userPrompt,
    },
  ];

  let iterations = 0;
  let lastOutput: OutputItems[] = [];

  while (iterations < maxIterations) {
    iterations += 1;

    lastOutput = await callLLM(client, conversationHistory, availableTools, {
      model,
      stream,
      onTextDelta: options.onTextDelta,
      onToolCall: options.onToolCall,
    });

    const toolCalls = lastOutput.filter(
      (item): item is OutputFunctionCallItem => item.type === "function_call",
    );

    if (toolCalls.length === 0) {
      return {
        status: "completed",
        iterations,
        text: extractFinalText(lastOutput),
      };
    }

    const toolOutputs = await Promise.all(
      toolCalls.map((toolCall) => executeToolCall(toolCall, runSubagent)),
    );
    conversationHistory.push(...toolOutputs);
  }

  return {
    status: "max_iterations",
    iterations,
    text: extractFinalText(lastOutput),
  };
}
