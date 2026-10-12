import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { relative } from 'node:path';
import { argv, cwd, stdin, stdout } from 'node:process';
import { createInterface } from 'node:readline';

import { chunkText } from './chunker.js';
import {
  CANDIDATES,
  CHUNK_OVERLAP_LINES,
  CHUNK_SIZE,
  EMBEDDING_MODEL,
  INDEX_PATH,
  RRF_K,
  SOURCE_PATH,
  TOP_K,
  redact,
} from './config.js';
import { embed } from './embeddings.js';
import { reciprocalRankFusion } from './hybrid-search.js';
import { buildKeywordIndex, keywordSearch } from './keyword-search.js';
import { loadIndex, saveIndex, search } from './vector-store.js';

const MODES = ['hybrid', 'vector', 'keyword'];

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

async function vectorSearch(index, query, topK) {
  const [queryVector] = await embed([query]);
  return search(index, queryVector, topK);
}

async function retrieve({ index, kwIndex }, query, mode) {
  if (mode === 'keyword') return keywordSearch(kwIndex, query, TOP_K);
  if (mode === 'vector') return vectorSearch(index, query, TOP_K);

  const [vector, keyword] = [
    await vectorSearch(index, query, CANDIDATES),
    keywordSearch(kwIndex, query, CANDIDATES),
  ];
  return reciprocalRankFusion({ vector, keyword }, RRF_K).slice(0, TOP_K);
}

function describe(result, mode) {
  if (mode !== 'hybrid') return `score ${result.score.toFixed(4)}`;
  const part = (name, hit, digits) =>
    hit ? `${name} #${hit.rank} (${hit.score.toFixed(digits)})` : `${name} —`;
  return [
    `rrf ${result.score.toFixed(4)}`,
    part('vector', result.vector, 4),
    part('keyword', result.keyword, 2),
  ].join('  | ');
}

async function runQuery(indexes, query, mode) {
  const results = await retrieve(indexes, query, mode);

  console.log(`\n[${mode}] Top ${results.length} results for: "${query}"\n`);
  if (!results.length) console.log('    (no matches)\n');
  for (const [rank, result] of results.entries()) {
    console.log(`#${rank + 1}  ${describe(result, mode)}  (chunk ${result.id})`);
    console.log(result.text.replace(/^/gm, '    '));
    console.log();
  }
}

async function interactive(indexes, initialMode) {
  let mode = initialMode;
  const rl = createInterface({ input: stdin, output: stdout, prompt: '\n> ' });
  console.log(`Loaded ${indexes.index.chunks.length} chunks. Mode: ${mode}.`);
  console.log(`Ask a question, switch with "/mode ${MODES.join('|')}", empty line or "exit" to quit.`);

  let closed = false;
  rl.on('close', () => (closed = true));

  rl.prompt();
  // for-await ends cleanly on Ctrl+C / end of input as well.
  for await (const line of rl) {
    const input = line.trim();
    if (!input || input === 'exit') break;

    const modeSwitch = input.match(/^\/mode\s+(\S+)$/);
    if (modeSwitch) {
      try {
        mode = parseMode(modeSwitch[1]);
        console.log(`Mode: ${mode}`);
      } catch (err) {
        console.log(err.message);
      }
    } else {
      await runQuery(indexes, input, mode);
    }
    if (!closed) rl.prompt();
  }
  rl.close();
}

function parseMode(value) {
  if (!MODES.includes(value)) throw new Error(`Unknown mode "${value}". Use one of: ${MODES.join(', ')}`);
  return value;
}

async function main(args) {
  const modeFlag = args.find((arg) => arg.startsWith('--mode='));
  const mode = parseMode(modeFlag ? modeFlag.slice('--mode='.length) : 'hybrid');
  const [command, ...rest] = args.filter((arg) => arg !== modeFlag);

  if (command === 'index') {
    await buildIndex();
    return;
  }

  const index = await getIndex();
  const indexes = { index, kwIndex: buildKeywordIndex(index.chunks) };

  if (command === 'search') {
    const query = rest.join(' ').trim();
    if (!query) throw new Error('Usage: npm run search -- "your question"');
    await runQuery(indexes, query, mode);
    return;
  }

  await interactive(indexes, mode);
}

main(argv.slice(2)).catch((err) => {
  const status = err.status ? ` [${err.status}${err.code ? ` ${err.code}` : ''}]` : '';
  console.error(`Error${status}: ${redact(err.message)}`);
  process.exitCode = 1;
});
