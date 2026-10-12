import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseEnv } from 'node:util';

const fromRoot = (relativePath) => fileURLToPath(new URL(`../${relativePath}`, import.meta.url));

// Load .env manually (instead of process.loadEnvFile) so a UTF-8 BOM
// saved by Windows editors doesn't corrupt the first variable name.
const ENV_PATH = fromRoot('.env');
if (existsSync(ENV_PATH)) {
  const vars = parseEnv(readFileSync(ENV_PATH, 'utf8').replace(/^﻿/, ''));
  for (const [name, value] of Object.entries(vars)) {
    process.env[name] ??= value;
  }
}

export const SOURCE_PATH = fromRoot('doc/context.txt');
export const INDEX_PATH = fromRoot('data/index.json');

export const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL ?? 'text-embedding-3-small';
export const CHUNK_SIZE = 500;
export const CHUNK_OVERLAP_LINES = 1;
export const TOP_K = 3;

// Strips anything that looks like an OpenAI key (even partially masked)
// from messages, so errors are safe to show on a shared screen.
export const redact = (message) => String(message).replace(/sk-[A-Za-z0-9_\-*]+/g, '[redacted]');

// Never print the key itself, only whether it exists.
export function assertApiKey() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set in .env');
  }
}
