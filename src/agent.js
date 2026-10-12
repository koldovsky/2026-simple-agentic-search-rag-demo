// Hand-written agentic RAG loop on top of the OpenAI Responses API.
//
//   question ─► LLM ─► search_document(query) ─► results ─► LLM ─► ... ─► answer
//
// The LLM decides whether the results answer the question; if not, it searches
// again with a new query. The loop enforces the search budget, not the model.

import { AGENT_TOP_K, LLM_MODEL, LLM_REASONING_EFFORT, MAX_SEARCHES, MAX_STEPS } from './config.js';
import { embed } from './embeddings.js';
import { getClient } from './openai-client.js';
import { searchDb } from './search-engine.js';
import { SYSTEM_PROMPT } from './system-prompt.js';

const SEARCH_TOOL = {
  type: 'function',
  name: 'search_document',
  description:
    'Hybrid (keyword + semantic) search over the IT STEP University document. ' +
    'Returns the most relevant text chunks you have not seen yet.',
  strict: true,
  parameters: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Search query, preferably in Ukrainian like the document.' },
    },
    required: ['query'],
    additionalProperties: false,
  },
};

const BUDGET_EXHAUSTED_NOTE =
  'Search budget is used up. Answer now using only the results above, ' +
  "or say the document doesn't contain this information.";

// Runs one search and returns only chunks the agent hasn't been shown yet.
async function searchNewChunks(db, query, seen) {
  const [vector] = await embed([query]);
  const hits = await searchDb(db, { mode: 'hybrid', term: query, vector, limit: AGENT_TOP_K + seen.size });
  const fresh = hits.filter((hit) => !seen.has(hit.id)).slice(0, AGENT_TOP_K);
  fresh.forEach((hit) => seen.add(hit.id));
  return fresh;
}

export async function runAgent(db, question, { onEvent = () => {} } = {}) {
  const client = getClient();
  const input = [{ role: 'user', content: question }];
  const seen = new Set();
  const usage = { llmCalls: 0, tokens: 0 };
  let searches = 0;

  for (let step = 0; step < MAX_STEPS; step++) {
    const budgetLeft = searches < MAX_SEARCHES;
    const response = await client.responses.create({
      model: LLM_MODEL,
      instructions: SYSTEM_PROMPT,
      input,
      tools: [SEARCH_TOOL],
      tool_choice: budgetLeft ? 'auto' : 'none',
      parallel_tool_calls: false,
      reasoning: { effort: LLM_REASONING_EFFORT },
    });
    usage.llmCalls++;
    usage.tokens += response.usage?.total_tokens ?? 0;

    // Keep the model's own output (reasoning + tool calls) in the transcript.
    input.push(...response.output);

    const calls = response.output.filter((item) => item.type === 'function_call');
    if (!calls.length) {
      return { answer: response.output_text, searches, usage };
    }

    for (const call of calls) {
      const { query } = JSON.parse(call.arguments);
      searches++;
      const results = await searchNewChunks(db, query, seen);
      onEvent({ type: 'search', n: searches, max: MAX_SEARCHES, query, chunkIds: results.map((r) => r.id) });

      input.push({
        type: 'function_call_output',
        call_id: call.call_id,
        output: JSON.stringify({
          results: results.map((r) => ({ chunk_id: r.id, text: r.text })),
          searches_left: Math.max(0, MAX_SEARCHES - searches),
        }),
      });
    }

    if (searches >= MAX_SEARCHES) {
      onEvent({ type: 'budget' });
      input.push({ role: 'developer', content: BUDGET_EXHAUSTED_NOTE });
    }
  }

  return {
    answer: "Sorry, I couldn't finish answering this question. Please contact the admissions office.",
    searches,
    usage,
  };
}
