import OpenAI from 'openai';
import { assertApiKey } from './config.js';

let client;

// Created lazily so commands that never call the API (e.g. keyword search) work without a key.
export function getClient() {
  if (!client) {
    assertApiKey();
    // The SDK reads OPENAI_API_KEY from process.env by itself.
    client = new OpenAI();
  }
  return client;
}
