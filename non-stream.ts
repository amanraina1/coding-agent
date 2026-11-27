import {
  DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT,
  DEFAULT_ORCHESTRATOR_USER_PROMPT,
  runAgentLoop,
} from "./agent.ts";

async function main() {
  const cliPrompt = process.argv.slice(2).join(" ").trim();
  const userPrompt = cliPrompt || DEFAULT_ORCHESTRATOR_USER_PROMPT;

  const result = await runAgentLoop({
    systemPrompt: DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT,
    userPrompt,
    stream: false,
    onToolCall: (toolCall) => {
      console.log(`\n[Calling tool: ${toolCall.name}(${toolCall.arguments})]`);
    },
  });

  if (result.status === "max_iterations") {
    console.warn(`Warning: Maximum iterations reached (${result.iterations})`);
  }

  console.log(result.text);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
