// Reciprocal Rank Fusion: merges ranked lists using only positions,
// so cosine scores (~0.2–0.7) and BM25 scores (0–15+) never need normalizing.
// score(chunk) = Σ 1 / (k + rank) over every list the chunk appears in.
export function reciprocalRankFusion(lists, k = 60) {
  const merged = new Map();

  for (const [source, results] of Object.entries(lists)) {
    results.forEach((result, i) => {
      const rank = i + 1;
      const entry = merged.get(result.id) ?? {
        id: result.id,
        text: result.text,
        score: 0,
        ...Object.fromEntries(Object.keys(lists).map((name) => [name, null])),
      };
      entry.score += 1 / (k + rank);
      entry[source] = { rank, score: result.score };
      merged.set(result.id, entry);
    });
  }

  return [...merged.values()].sort((a, b) => b.score - a.score);
}
