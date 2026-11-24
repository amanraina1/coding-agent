import { OpenRouter } from '@openrouter/sdk';
import { callModel, tool } from '@openrouter/agent';
import { z } from 'zod';
import axios from "axios"

const client = new OpenRouter({ apiKey: process.env.API_KEY });

const fetchWeather = async (query: string) => {
    const response = await axios.get("http://api.weatherapi.com/v1/current.json", {
        params: {
            key: process.env.WEATHER_API_KEY,
            q: query
        }
    });

    const resposneObj = {
        temperature: String(response.data.location.name),
        conditions: String(response.data.current.condition.text)
    }

    return resposneObj
}

const weatherTool = tool({
  name: 'get_weather',
  description: 'Get the current weather in a location',
  inputSchema: z.object({
    location: z.string(),
    unit: z.string()
  }),
  outputSchema: z.object({
    temperature: z.string().nullable(),
    conditions: z.string().nullable(),
  }),
  execute: async ({ location, unit }) => await fetchWeather(location),
});

const result = callModel(client, {
  model: 'openai/gpt-4o-mini',
  input: 'What is the weather like in Gurgaon?',
  tools: [weatherTool] as const,
});

// Tools are executed automatically; this is the model's final answer.
console.log(await result.getText());