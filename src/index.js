import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Command, Option } from 'commander';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { relative } from 'node:path';
import { cwd, stdin, stdout } from 'node:process';
import { createInterface } from 'node:readline';

import {
  CHUNK_OVERLAP,
  CHUNK_SIZE,
  EMBEDDING_MODEL,
  INDEX_PATH,
  SOURCE_PATH,
  TOP_K,
  redact,
} from './config.js';
import { embed } from './embeddings.js';
import { buildDb, loadIndex, saveIndex, searchDb, totalChunks } from './search-engine.js';

// CLI mode name -> Orama search mode
const MODES = { hybrid: 'hybrid', vector: 'vector', keyword: 'fulltext' };

const sha256 = (text) => createHash('sha256').update(text).digest('hex');

async function buildIndex() {
  const source = await readFile(SOURCE_PATH, 'utf8');
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: CHUNK_SIZE, chunkOverlap: CHUNK_OVERLAP });
  // The doc has CRLF line endings and trailing spaces; tidy them before splitting.
  const chunks = await splitter.splitText(source.replace(/[ \t]+$/gm, '').replace(/\r\n?/g, '\n'));

  console.log(`Embedding ${chunks.length} chunks with ${EMBEDDING_MODEL}...`);
  const vectors = await embed(chunks);
  const db = await buildDb(chunks, vectors);

  const meta = {
    model: EMBEDDING_MODEL,
    dimensions: vectors[0].length,
    sourceHash: sha256(source),
    createdAt: new Date().toISOString(),
    chunkCount: chunks.length,
  };
  await saveIndex(INDEX_PATH, db, meta);
  console.log(`Saved ${chunks.length} chunks to ${relative(cwd(), INDEX_PATH)}`);
  return db;
}

// Reuses the stored index unless the source doc or the model changed.
async function getDb() {
  const index = await loadIndex(INDEX_PATH);
  const source = await readFile(SOURCE_PATH, 'utf8');
  if (index && index.meta.model === EMBEDDING_MODEL && index.meta.sourceHash === sha256(source)) {
    return index.db;
  }
  console.log('Index is missing or stale, rebuilding...');
  return buildIndex();
}

const rankMap = (results) => new Map(results.map((result, i) => [result.id, { rank: i + 1, score: result.score }]));

async function retrieve(db, query, mode) {
  const oramaMode = MODES[mode];
  // Keyword mode never calls the API, so it works offline.
  const vector = oramaMode === 'fulltext' ? undefined : (await embed([query]))[0];
  const results = await searchDb(db, { mode: oramaMode, term: query, vector, limit: TOP_K });
  if (mode !== 'hybrid') return results;

  // For the demo: show where each retriever ranked every hybrid hit.
  const limit = totalChunks(db);
  const vectorRanks = rankMap(await searchDb(db, { mode: 'vector', vector, limit }));
  const keywordRanks = rankMap(await searchDb(db, { mode: 'fulltext', term: query, limit }));
  return results.map((result) => ({
    ...result,
    vector: vectorRanks.get(result.id) ?? null,
    keyword: keywordRanks.get(result.id) ?? null,
  }));
}

function describe(result, mode) {
  if (mode !== 'hybrid') return `score ${result.score.toFixed(4)}`;
  const part = (name, hit, digits) =>
    hit ? `${name} #${hit.rank} (${hit.score.toFixed(digits)})` : `${name} —`;
  return [
    `hybrid ${result.score.toFixed(4)}`,
    part('vector', result.vector, 4),
    part('keyword', result.keyword, 2),
  ].join('  | ');
}

async function runQuery(db, query, mode) {
  const results = await retrieve(db, query, mode);

  console.log(`\n[${mode}] Top ${results.length} results for: "${query}"\n`);
  if (!results.length) console.log('    (no matches)\n');
  for (const [rank, result] of results.entries()) {
    console.log(`#${rank + 1}  ${describe(result, mode)}  (chunk ${result.id})`);
    console.log(result.text.replace(/^/gm, '    '));
    console.log();
  }
}

async function interactive(db, initialMode) {
  let mode = initialMode;
  const modeNames = Object.keys(MODES);
  const rl = createInterface({ input: stdin, output: stdout, prompt: '\n> ' });
  console.log(`Loaded ${totalChunks(db)} chunks. Mode: ${mode}.`);
  console.log(`Ask a question, switch with "/mode ${modeNames.join('|')}", empty line or "exit" to quit.`);

  let closed = false;
  rl.on('close', () => (closed = true));

  rl.prompt();
  // for-await ends cleanly on Ctrl+C / end of input as well.
  for await (const line of rl) {
    const input = line.trim();
    if (!input || input === 'exit') break;

    const modeSwitch = input.match(/^\/mode\s+(\S+)$/);
    if (modeSwitch) {
      if (modeSwitch[1] in MODES) {
        mode = modeSwitch[1];
        console.log(`Mode: ${mode}`);
      } else {
        console.log(`Unknown mode "${modeSwitch[1]}". Use one of: ${modeNames.join(', ')}`);
      }
    } else {
      await runQuery(db, input, mode);
    }
    if (!closed) rl.prompt();
  }
  rl.close();
}

const program = new Command()
  .name('hybrid-search')
  .description('Hybrid (BM25 + embeddings) search over doc/context.txt')
  .addOption(new Option('-m, --mode <mode>', 'search mode').choices(Object.keys(MODES)).default('hybrid'))
  .action(async () => interactive(await getDb(), program.opts().mode));

program
  .command('index')
  .description('chunk the doc, embed it and save data/index.json')
  .action(async () => {
    await buildIndex();
  });

program
  .command('search')
  .description('run a single query')
  .argument('<query...>', 'question to search for')
  .action(async (words, _options, command) => {
    await runQuery(await getDb(), words.join(' '), command.optsWithGlobals().mode);
  });

program.parseAsync().catch((err) => {
  const status = err.status ? ` [${err.status}${err.code ? ` ${err.code}` : ''}]` : '';
  console.error(`Error${status}: ${redact(err.message)}`);
  process.exitCode = 1;
});
