import OpenAI from 'openai';
import { EMBEDDING_MODEL, assertApiKey } from './config.js';

let client;

function getClient() {
  if (!client) {
    assertApiKey();
    // The SDK reads OPENAI_API_KEY from process.env by itself.
    client = new OpenAI();
  }
  return client;
}

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
