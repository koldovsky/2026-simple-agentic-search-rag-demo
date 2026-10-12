// Hand-written BM25 keyword search.

const STOP_WORDS = new Set([
  'і', 'й', 'та', 'в', 'у', 'на', 'з', 'зі', 'із', 'до', 'що', 'як', 'де', 'чи', 'є', 'для',
  'це', 'а', 'о', 'по', 'від', 'за', 'про', 'не', 'але', 'або', 'ж', 'же', 'ви', 'ми', 'вас',
  'вам', 'ваш', 'наш', 'наші', 'які', 'який', 'яка', 'яке', 'коли', 'скільки', 'чому', 'хто',
  'при', 'під', 'над', 'без', 'також', 'так', 'ще', 'вже', 'буде', 'бути', 'можна',
]);

const STEM_LENGTH = 5;
const K1 = 1.2;
const B = 0.75;

// Lowercase, drop apostrophes, split on non-letters/digits, remove stop words,
// then crude prefix stemming so Ukrainian word forms match each other
// (гуртожиток / гуртожитки -> "гурто", комісія / комісії -> "коміс").
export function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/['’ʼ`]/g, '')
    .split(/[^\p{L}\p{N}]+/u)
    .filter((token) => token.length >= 2 && !STOP_WORDS.has(token))
    .map((token) => token.slice(0, STEM_LENGTH));
}

export function buildKeywordIndex(chunks) {
  const df = new Map();
  const docs = chunks.map((chunk) => {
    const tokens = tokenize(chunk.text);
    const tf = new Map();
    for (const token of tokens) tf.set(token, (tf.get(token) ?? 0) + 1);
    for (const token of tf.keys()) df.set(token, (df.get(token) ?? 0) + 1);
    return { id: chunk.id, text: chunk.text, tf, length: tokens.length };
  });

  const avgLength = docs.reduce((sum, doc) => sum + doc.length, 0) / docs.length;
  return { docs, df, avgLength, count: docs.length };
}

export function keywordSearch(kwIndex, query, topK) {
  const queryTokens = [...new Set(tokenize(query))];
  const { docs, df, avgLength, count } = kwIndex;

  return docs
    .map((doc) => {
      let score = 0;
      for (const token of queryTokens) {
        const freq = doc.tf.get(token);
        if (!freq) continue;
        const docFreq = df.get(token);
        const idf = Math.log(1 + (count - docFreq + 0.5) / (docFreq + 0.5));
        score += idf * (freq * (K1 + 1)) / (freq + K1 * (1 - B + B * (doc.length / avgLength)));
      }
      return { id: doc.id, text: doc.text, score };
    })
    .filter((result) => result.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
