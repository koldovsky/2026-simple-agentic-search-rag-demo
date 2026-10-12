import { count, create, insertMultiple, load, save, search } from '@orama/orama';
import { language, stemmer } from '@orama/stemmers/ukrainian';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
// @orama/stopwords/ukrainian ships a mis-encoded list, so use `stopword`'s instead.
import { ukr as ukrainianStopWords } from 'stopword';

import { HYBRID_WEIGHTS, KEYWORD_TOLERANCE, MIN_SIMILARITY } from './config.js';

// The Ukrainian tokenizer is configured in code, not stored in the JSON,
// so a loaded index must be created through here to keep stemming intact.
function createDb(dimensions) {
  return create({
    schema: {
      chunkId: 'number',
      text: 'string',
      embedding: `vector[${dimensions}]`,
    },
    components: {
      tokenizer: { language, stemming: true, stemmer, stopWords: ukrainianStopWords },
    },
  });
}

export async function buildDb(chunks, vectors) {
  const db = createDb(vectors[0].length);
  await insertMultiple(
    db,
    chunks.map((text, i) => ({ id: String(i), chunkId: i, text, embedding: vectors[i] })),
  );
  return db;
}

export async function saveIndex(path, db, meta) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify({ ...meta, orama: save(db) }), 'utf8');
}

export async function loadIndex(path) {
  if (!existsSync(path)) return null;
  const { orama, ...meta } = JSON.parse(await readFile(path, 'utf8'));
  const db = createDb(meta.dimensions);
  load(db, orama);
  return { db, meta };
}

export const totalChunks = (db) => count(db);

// mode: 'fulltext' (BM25) | 'vector' | 'hybrid'
export async function searchDb(db, { mode, term, vector, limit }) {
  const params = { mode, limit };
  if (mode !== 'vector') {
    Object.assign(params, { term, properties: ['text'], tolerance: KEYWORD_TOLERANCE });
  }
  if (mode !== 'fulltext') {
    Object.assign(params, {
      vector: { value: vector, property: 'embedding' },
      similarity: MIN_SIMILARITY,
    });
  }
  if (mode === 'hybrid') params.hybridWeights = HYBRID_WEIGHTS;

  const { hits } = await search(db, params);
  return hits.map((hit) => ({ id: hit.document.chunkId, text: hit.document.text, score: hit.score }));
}
