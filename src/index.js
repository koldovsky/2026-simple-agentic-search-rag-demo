import { Command, Option } from 'commander';
import { stdin, stdout } from 'node:process';
import { createInterface } from 'node:readline';

import { runAgent } from './agent.js';
import { LLM_MODEL, TOP_K, redact } from './config.js';
import { embed } from './embeddings.js';
import { buildIndex, getDb } from './indexer.js';
import { searchDb, totalChunks } from './search-engine.js';

// CLI mode name -> Orama search mode
const MODES = { hybrid: 'hybrid', vector: 'vector', keyword: 'fulltext' };

// ---------- agent ----------

function printEvent(event) {
  if (event.type === 'search') {
    const found = event.chunkIds.length ? `chunks ${event.chunkIds.join(', ')}` : 'no new chunks';
    console.log(`🔎 search ${event.n}/${event.max}: "${event.query}" → ${found}`);
  } else if (event.type === 'budget') {
    console.log('⛔ search budget used up, answering with what was found');
  }
}

async function ask(db, question) {
  const { answer, searches, usage } = await runAgent(db, question, { onEvent: printEvent });
  const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  const stats = [
    count(searches, 'search', 'searches'),
    count(usage.llmCalls, 'LLM call', 'LLM calls'),
    `${(usage.tokens / 1000).toFixed(1)}k tokens`,
  ];
  console.log(`💬 answer (${stats.join(', ')})\n`);
  console.log(answer);
}

async function chat(db) {
  const rl = createInterface({ input: stdin, output: stdout, prompt: '\n> ' });
  console.log(`Agentic RAG over ${totalChunks(db)} chunks, model ${LLM_MODEL}.`);
  console.log('Ask a question about IT STEP University (empty line or "exit" to quit).');

  let closed = false;
  rl.on('close', () => (closed = true));

  rl.prompt();
  // for-await ends cleanly on Ctrl+C / end of input as well.
  for await (const line of rl) {
    const question = line.trim();
    if (!question || question === 'exit') break;
    try {
      await ask(db, question);
    } catch (err) {
      printError(err);
    }
    if (!closed) rl.prompt();
  }
  rl.close();
}

// ---------- raw retrieval (for comparison) ----------

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

// ---------- CLI ----------

function printError(err) {
  const status = err.status ? ` [${err.status}${err.code ? ` ${err.code}` : ''}]` : '';
  console.error(`Error${status}: ${redact(err.message)}`);
}

const program = new Command()
  .name('agentic-rag')
  .description('Agentic search RAG over doc/context.txt')
  .action(async () => chat(await getDb()));

program
  .command('ask')
  .description('ask the agent a single question')
  .argument('<question...>', 'question about the document')
  .action(async (words) => ask(await getDb(), words.join(' ')));

program
  .command('index')
  .description('chunk the doc, embed it and save data/index.json')
  .action(async () => {
    await buildIndex();
  });

program
  .command('search')
  .description('raw retrieval without the agent (for comparison)')
  .argument('<query...>', 'query to search for')
  .addOption(new Option('-m, --mode <mode>', 'search mode').choices(Object.keys(MODES)).default('hybrid'))
  .action(async (words, options) => runQuery(await getDb(), words.join(' '), options.mode));

program.parseAsync().catch((err) => {
  printError(err);
  process.exitCode = 1;
});
