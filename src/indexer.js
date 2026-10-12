import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { relative } from 'node:path';
import { cwd } from 'node:process';

import { CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, INDEX_PATH, SOURCE_PATH } from './config.js';
import { embed } from './embeddings.js';
import { buildDb, loadIndex, saveIndex } from './search-engine.js';

const sha256 = (text) => createHash('sha256').update(text).digest('hex');

export async function buildIndex() {
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
export async function getDb() {
  const index = await loadIndex(INDEX_PATH);
  const source = await readFile(SOURCE_PATH, 'utf8');
  if (index && index.meta.model === EMBEDDING_MODEL && index.meta.sourceHash === sha256(source)) {
    return index.db;
  }
  console.log('Index is missing or stale, rebuilding...');
  return buildIndex();
}
