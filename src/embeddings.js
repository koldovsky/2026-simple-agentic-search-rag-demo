import { EMBEDDING_MODEL } from './config.js';
import { getClient } from './openai-client.js';

// Embeds a batch of texts in one request; returns vectors in input order.
export async function embed(texts) {
  const response = await getClient().embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts,
  });
  return response.data
    .sort((a, b) => a.index - b.index)
    .map((item) => item.embedding);
}
