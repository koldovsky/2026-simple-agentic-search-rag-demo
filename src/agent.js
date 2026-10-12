// Agentic RAG on top of the Vercel AI SDK's ToolLoopAgent.
//
//   question ─► LLM ─► search_document(query) ─► results ─► LLM ─► ... ─► answer
//
// The LLM decides whether the results answer the question; if not, it searches
// again with a new query. The search budget is enforced in code (prepareStep),
// not left to the model.

import { openai } from '@ai-sdk/openai';
import { isStepCount, tool, ToolLoopAgent } from 'ai';
import { z } from 'zod';

import {
  AGENT_TOP_K,
  LLM_MODEL,
  LLM_REASONING_EFFORT,
  MAX_SEARCHES,
  MAX_STEPS,
  assertApiKey,
} from './config.js';
import { embed } from './embeddings.js';
import { searchDb } from './search-engine.js';
import { SYSTEM_PROMPT } from './system-prompt.js';

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
  assertApiKey();

  // Per-question state, shared by the tool and prepareStep through this closure.
  const seen = new Set();
  let searches = 0;
  let budgetAnnounced = false;

  const searchDocument = tool({
    description:
      'Hybrid (keyword + semantic) search over the IT STEP University document. ' +
      'Returns the most relevant text chunks you have not seen yet.',
    inputSchema: z.object({
      query: z.string().describe('Search query, preferably in Ukrainian like the document.'),
    }),
    execute: async ({ query }) => {
      searches++;
      const results = await searchNewChunks(db, query, seen);
      onEvent({ type: 'search', n: searches, max: MAX_SEARCHES, query, chunkIds: results.map((r) => r.id) });
      return {
        results: results.map((r) => ({ chunk_id: r.id, text: r.text })),
        searches_left: Math.max(0, MAX_SEARCHES - searches),
      };
    },
  });

  const agent = new ToolLoopAgent({
    model: openai(LLM_MODEL), // Responses API
    instructions: SYSTEM_PROMPT,
    tools: { search_document: searchDocument },
    // Hard cap on LLM calls per question.
    stopWhen: isStepCount(MAX_STEPS),
    // Before every LLM call: once the budget is spent, take the tool away
    // so the model has to answer with what it found (or give up).
    prepareStep: () => {
      if (searches < MAX_SEARCHES) return {};
      if (!budgetAnnounced) {
        budgetAnnounced = true;
        onEvent({ type: 'budget' });
      }
      return {
        activeTools: [],
        toolChoice: 'none',
        instructions: `${SYSTEM_PROMPT}\n\n${BUDGET_EXHAUSTED_NOTE}`,
      };
    },
    providerOptions: {
      openai: { reasoningEffort: LLM_REASONING_EFFORT, parallelToolCalls: false },
    },
  });

  const result = await agent.generate({ prompt: question });
  return {
    answer: result.text,
    searches,
    usage: { llmCalls: result.steps.length, tokens: result.totalUsage.totalTokens ?? 0 },
  };
}
