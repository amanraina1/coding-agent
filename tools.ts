import type {
  FunctionCallOutputItem,
  OutputFunctionCallItem,
  ResponsesRequestToolFunction,
} from "@openrouter/sdk/models";
import axios from "axios";
import { exec } from "node:child_process";
import { promisify } from "node:util";

export interface SubagentRequest {
  task: string;
  role?: string;
  context?: string;
}

export interface SubagentResult {
  status: "completed" | "max_iterations" | "error";
  role: string;
  iterations: number;
  result: string;
}

export type SubagentRunner = (request: SubagentRequest) => Promise<SubagentResult>;

const weatherTool: ResponsesRequestToolFunction = {
  type: "function",
  name: "get_weather",
  description: "Get the current weather in a location",
  parameters: {
    type: "object",
    properties: {
      location: {
        type: "string",
        description: "Name of the location for which the weather data needs to be fetched",
      },
    },
    required: ["location"],
  },
};

const bashTool: ResponsesRequestToolFunction = {
  type: "function",
  name: "execute_bash",
  description:
    "Execute a bash command in the current working directory and return its exit code, stdout, and stderr",
  parameters: {
    type: "object",
    properties: {
      command: {
        type: "string",
        description: "Command to execute",
      },
    },
    required: ["command"],
  },
};

export const spawnSubagentTool: ResponsesRequestToolFunction = {
  type: "function",
  name: "spawn_subagent",
  description:
    "Spawn an independent subagent to handle a focused subtask and return its final result. " +
    "Use this when a task is large or can be broken into independent parts. " +
    "Each subagent starts with a fresh conversation and has access to get_weather and execute_bash, " +
    "but cannot spawn further subagents. You may call this tool multiple times in the same turn to run subagents in parallel.",
  parameters: {
    type: "object",
    properties: {
      task: {
        type: "string",
        description:
          "Clear, self-contained instructions for the subagent, including what final answer or findings it should return.",
      },
      role: {
        type: "string",
        description:
          'Optional role or specialization for the subagent (e.g., "Codebase Researcher", "Shell Worker").',
      },
      context: {
        type: "string",
        description:
          "Optional background context, file paths, or constraints needed because the subagent cannot see the parent conversation.",
      },
    },
    required: ["task"],
  },
};

export const workerTools: ResponsesRequestToolFunction[] = [weatherTool, bashTool];
export const orchestratorTools: ResponsesRequestToolFunction[] = [...workerTools, spawnSubagentTool];

/**
 * Returns the toolset permitted at the given recursion depth.
 */
export function getToolsForDepth(depth: number, maxDepth: number): ResponsesRequestToolFunction[] {
  return depth < maxDepth ? orchestratorTools : workerTools;
}

export async function fetchWeather(query: string) {
  const response = await axios.get("http://api.weatherapi.com/v1/current.json", {
    params: {
      key: process.env.WEATHER_API_KEY,
      q: query,
    },
  });

  return {
    location: String(response.data.location.name),
    temperature: `${response.data.current.temp_c}°C`,
    conditions: String(response.data.current.condition.text),
  };
}

const execAsync = promisify(exec);

export async function executeBash(command: string): Promise<string> {
  try {
    const { stdout, stderr } = await execAsync(command, {
      timeout: 20000,
      maxBuffer: 1024 * 1024,
    });

    let output = "Exit Code: 0\n";
    if (stdout) output += `STDOUT:\n${stdout}\n`;
    if (stderr) output += `STDERR:\n${stderr}\n`;
    return output.trim();
  } catch (error) {
    const err = error as {
      code?: number | string;
      stdout?: string;
      stderr?: string;
      message?: string;
    };
    let output = `Exit Code: ${err.code ?? 1}\n`;
    if (err.stdout) output += `STDOUT:\n${err.stdout}\n`;
    if (err.stderr) output += `STDERR:\n${err.stderr}\n`;
    if (!err.stdout && !err.stderr && err.message) {
      output += `ERROR:\n${err.message}\n`;
    }
    return output.trim();
  }
}

function parseToolArgs(rawArguments: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(rawArguments);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("Tool arguments must be a JSON object");
  }
  return parsed as Record<string, unknown>;
}

/**
 * Executes a single model tool call and formats the result as a FunctionCallOutputItem.
 */
export async function executeToolCall(
  toolCall: OutputFunctionCallItem,
  runSubagent?: SubagentRunner,
): Promise<FunctionCallOutputItem> {
  let toolResult: unknown;

  try {
    const args = parseToolArgs(toolCall.arguments);

    switch (toolCall.name) {
      case "execute_bash": {
        if (typeof args.command !== "string" || args.command.trim() === "") {
          throw new Error('execute_bash requires a non-empty string "command"');
        }
        toolResult = await executeBash(args.command);
        break;
      }

      case "get_weather": {
        if (typeof args.location !== "string" || args.location.trim() === "") {
          throw new Error('get_weather requires a non-empty string "location"');
        }
        toolResult = await fetchWeather(args.location);
        break;
      }

      case "spawn_subagent": {
        if (!runSubagent) {
          toolResult = {
            status: "error",
            error: "spawn_subagent is not permitted at this agent depth.",
          };
          break;
        }
        if (typeof args.task !== "string" || args.task.trim() === "") {
          throw new Error('spawn_subagent requires a non-empty string "task"');
        }

        const request: SubagentRequest = {
          task: args.task,
          role: typeof args.role === "string" && args.role.trim() !== "" ? args.role.trim() : undefined,
          context:
            typeof args.context === "string" && args.context.trim() !== ""
              ? args.context.trim()
              : undefined,
        };
        toolResult = await runSubagent(request);
        break;
      }

      default:
        toolResult = { error: `Unknown tool: ${toolCall.name}` };
    }
  } catch (error) {
    toolResult = {
      error: error instanceof Error ? error.message : String(error),
    };
  }

  return {
    type: "function_call_output",
    callId: toolCall.callId,
    output: JSON.stringify(toolResult),
  };
}
