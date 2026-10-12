import dotenv from 'dotenv';
import { fileURLToPath } from 'node:url';

const fromRoot = (relativePath) => fileURLToPath(new URL(`../${relativePath}`, import.meta.url));

// quiet: dotenv would otherwise print a banner line on every run.
dotenv.config({ path: fromRoot('.env'), quiet: true });

export const SOURCE_PATH = fromRoot('doc/context.txt');
export const INDEX_PATH = fromRoot('data/index.json');

export const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? 'text-embedding-3-small';
export const CHUNK_SIZE = 500;
export const CHUNK_OVERLAP = 60;
export const TOP_K = 3;
// Orama's default vector threshold is 0.8; our cosine scores sit around 0.2–0.65.
export const MIN_SIMILARITY = 0.1;
export const HYBRID_WEIGHTS = { text: 0.5, vector: 0.5 };
// Levenshtein distance 1 bridges stemmer gaps like гуртожиток / гуртожитки.
export const KEYWORD_TOLERANCE = 1;

// Agent
export const LLM_MODEL = process.env.LLM_MODEL ?? 'gpt-6.1-sol';
export const LLM_REASONING_EFFORT = process.env.LLM_REASONING_EFFORT ?? 'low';
// 1 initial search + 2 re-searches. Each search returns only chunks the agent hasn't
// seen yet, so 3 searches cover ~12 distinct chunks (~30% of the doc) with 3 phrasings;
// if hybrid search misses that often, the answer is almost certainly not in the doc.
export const MAX_SEARCHES = 3;
export const MAX_STEPS = 6; // hard cap on LLM calls per question
export const AGENT_TOP_K = 4;

// Strips anything that looks like an OpenAI key (even partially masked)
// from messages, so errors are safe to show on a shared screen.
export const redact = (message) => String(message).replace(/sk-[A-Za-z0-9_\-*]+/g, '[redacted]');

// Never print the key itself, only whether it exists.
export function assertApiKey() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set in .env');
  }
}
