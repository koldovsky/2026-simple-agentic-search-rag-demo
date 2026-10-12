import { openai } from '@ai-sdk/openai';
import { embedMany } from 'ai';

import { EMBEDDING_MODEL, assertApiKey } from './config.js';

// Embeds a batch of texts in one request; returns vectors in input order.
export async function embed(texts) {
  assertApiKey();
  const { embeddings } = await embedMany({
    model: openai.embedding(EMBEDDING_MODEL),
    values: texts,
  });
  return embeddings;
}
