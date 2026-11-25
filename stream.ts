import { OpenRouter } from "@openrouter/sdk"
import type { FunctionCallOutputItem, InputsUnion, OutputFunctionCallItem, ResponsesRequestToolFunction } from "@openrouter/sdk/models";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import axios from "axios";

const client = new OpenRouter({
    apiKey: process.env.API_KEY
});

const input: InputsUnion = [
    {
        content: "You are a helpful assistent, you have access to bunch of tools, use them to get data whatever you need. Remember the tool names perfectly. don't use any other name, other than what is specified here",
        role: "system"
    },
    { 
        // content: "Can you tell me the weather of today in Gurgaon and also list all the files and folders in this directory", 
        content: "Currently if you check in index.ts file i only have 2 tool calls integrated, and i want to add another tool call which is to implement multi agent orchestration, can you tell me what is multi agent orchestration and how can i implement that",
        role: "user" 
    }
];

const fetchWeather = async (query: string) => {
  const response = await axios.get("http://api.weatherapi.com/v1/current.json", {
    params: {
      key: process.env.WEATHER_API_KEY,
      q: query
    }
  });

  return {
    location: String(response.data.location.name),
    temperature: `${response.data.current.temp_c}°C`,
    conditions: String(response.data.current.condition.text)
  };
};

const execAsync = promisify(exec);
async function execute_bash(command: string) {
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
    return error
  }
}

const tools: ResponsesRequestToolFunction[] = [
  {
    type: 'function' as const,
    name: 'get_weather',
    description: 'Get the current weather in a location',
    parameters: {
      type: 'object',
      properties: {
        location: {
          type: 'string',
          description: "Name of the location for which the weather data needs to be fetched"
        }
      },
      required: ['location'],
    },
  },
  {
    type: 'function' as const,
    name: 'execute_bash',
    description: 'Execute a bash command to list files in the directory',
    parameters: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description: "Command to execute"
        }
      },
      required: ['command'],
    },
  },
];

async function getToolResponse(toolCall: OutputFunctionCallItem): Promise<FunctionCallOutputItem> {

  const args = JSON.parse(toolCall.arguments);

  let toolResult;

  switch (toolCall.name) {

    case "execute_bash":
        toolResult = await execute_bash(args.command);
        break;

    case "get_weather":
        toolResult = await fetchWeather(args.location);
        break;

    default: 
        throw new Error(`Unknown tool: ${toolCall.name}`);    
  }

  return {
    type: 'function_call_output',
    callId: toolCall.callId,
    output: JSON.stringify(toolResult),
  };
}

async function callLLM(conversationHistory: InputsUnion) {
  const result = await client.responses.send({
    responsesRequest: {
      stream: true,
      tools,
      model: 'openai/gpt-oss-20b',
      input: conversationHistory,
    },
  });

  if (!('output' in result)) {
    let completedResponse: any = null;
    const fallbackOutputItems: any[] = [];

    for await (const event of result) {

        switch (event.type) {

            case "response.failed":
                throw new Error(event.response?.error?.message ?? 'Response failed');
                
            case "response.incomplete":
                throw new Error(`Response incomplete: ${event.response ?? 'unknown reason'}`);

            case "response.output_text.delta":
                process.stdout.write(event.delta ?? '');
                break;

            case "response.output_item.done": {
                if (event.item) {
                    fallbackOutputItems.push(event.item);
                    if (event.item.type === 'function_call') {
                        console.log(`\n[Calling tool: ${event.item.name}(${event.item.arguments})]`);
                    }
                }
                break;
            }

            case "response.completed": {
                completedResponse = event.response;
                break;
            }
        }
    }

    if (!completedResponse) {
      throw new Error('Stream ended before receiving response.completed event');
    }

    const outputItems = completedResponse?.output ?? fallbackOutputItems;
    if (outputItems.length === 0) {
        throw new Error('Stream ended without producing any output items');
    }

    conversationHistory.push(...outputItems);

    return {output: outputItems}
  }
}

async function main() {

    while(true) {
        const response = await callLLM(input);

        console.log("\n[COMPLETED RESPONSE]: ", response?.output)

        const toolCalls = response?.output.filter(item => item.type === "function_call");

        if(toolCalls.length == 0) {
            return;
        }

        const toolOutputs = await Promise.all(toolCalls.map(async (toolCall: OutputFunctionCallItem) => {
            const response = await getToolResponse(toolCall)
            return response
        }));

        input.push(...toolOutputs);
    }
}

main();