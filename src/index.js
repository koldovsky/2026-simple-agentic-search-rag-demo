import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { relative } from 'node:path';
import { argv, cwd, stdin, stdout } from 'node:process';
import { createInterface } from 'node:readline';

import { chunkText } from './chunker.js';
import {
  CHUNK_OVERLAP_LINES,
  CHUNK_SIZE,
  EMBEDDING_MODEL,
  INDEX_PATH,
  SOURCE_PATH,
  TOP_K,
  redact,
} from './config.js';
import { embed } from './embeddings.js';
import { loadIndex, saveIndex, search } from './vector-store.js';

const sha256 = (text) => createHash('sha256').update(text).digest('hex');

async function buildIndex() {
  const source = await readFile(SOURCE_PATH, 'utf8');
  const chunks = chunkText(source, { maxChars: CHUNK_SIZE, overlapLines: CHUNK_OVERLAP_LINES });

  console.log(`Embedding ${chunks.length} chunks with ${EMBEDDING_MODEL}...`);
  const vectors = await embed(chunks.map((chunk) => chunk.text));

  const index = {
    model: EMBEDDING_MODEL,
    sourceHash: sha256(source),
    createdAt: new Date().toISOString(),
    chunks: chunks.map((chunk, i) => ({ ...chunk, embedding: vectors[i] })),
  };
  await saveIndex(INDEX_PATH, index);
  console.log(`Saved ${chunks.length} chunks to ${relative(cwd(), INDEX_PATH)}`);
  return index;
}

// Reuses the stored index unless the source doc or the model changed.
async function getIndex() {
  const index = await loadIndex(INDEX_PATH);
  const source = await readFile(SOURCE_PATH, 'utf8');
  if (index && index.model === EMBEDDING_MODEL && index.sourceHash === sha256(source)) {
    return index;
  }
  console.log('Index is missing or stale, rebuilding...');
  return buildIndex();
}

async function runQuery(index, query) {
  const [queryVector] = await embed([query]);
  const results = search(index, queryVector, TOP_K);

  console.log(`\nTop ${results.length} results for: "${query}"\n`);
  for (const [rank, result] of results.entries()) {
    console.log(`#${rank + 1}  score ${result.score.toFixed(4)}  (chunk ${result.id})`);
    console.log(result.text.replace(/^/gm, '    '));
    console.log();
  }
}

async function interactive(index) {
  const rl = createInterface({ input: stdin, output: stdout, prompt: '\n> ' });
  console.log(`Loaded ${index.chunks.length} chunks. Ask a question (empty line or "exit" to quit).`);
  let closed = false;
  rl.on('close', () => (closed = true));

  rl.prompt();
  // for-await ends cleanly on Ctrl+C / end of input as well.
  for await (const line of rl) {
    const query = line.trim();
    if (!query || query === 'exit') break;
    await runQuery(index, query);
    if (!closed) rl.prompt();
  }
  rl.close();
}

async function main([command, ...rest]) {
  if (command === 'index') {
    await buildIndex();
    return;
  }

  const index = await getIndex();
  if (command === 'search') {
    const query = rest.join(' ').trim();
    if (!query) throw new Error('Usage: npm run search -- "your question"');
    await runQuery(index, query);
    return;
  }

  await interactive(index);
}

main(argv.slice(2)).catch((err) => {
  const status = err.status ? ` [${err.status}${err.code ? ` ${err.code}` : ''}]` : '';
  console.error(`Error${status}: ${redact(err.message)}`);
  process.exitCode = 1;
});
