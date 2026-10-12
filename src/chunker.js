// Splits text into small, paragraph-aware chunks of at most `maxChars`.
// Each new chunk repeats the last `overlapLines` lines of the previous one
// so facts that sit on a boundary are not lost.

function splitLongPiece(piece, maxChars) {
  if (piece.length <= maxChars) return [piece];

  const lines = piece.split('\n');
  if (lines.length > 1) {
    return lines.flatMap((line) => splitLongPiece(line, maxChars));
  }

  const sentences = piece.match(/[^.!?]+[.!?]*\s*/g) ?? [piece];
  if (sentences.length > 1) {
    return sentences.flatMap((sentence) => splitLongPiece(sentence.trim(), maxChars));
  }

  // A single huge "sentence": hard cut as a last resort.
  const parts = [];
  for (let i = 0; i < piece.length; i += maxChars) {
    parts.push(piece.slice(i, i + maxChars));
  }
  return parts;
}

export function chunkText(text, { maxChars = 500, overlapLines = 1 } = {}) {
  const paragraphs = text
    .replace(/\r\n?/g, '\n')
    .split(/\n\s*\n/)
    .map((p) => p.split('\n').map((line) => line.trim()).filter(Boolean).join('\n'))
    .filter(Boolean);

  const pieces = paragraphs.flatMap((p) => splitLongPiece(p, maxChars));

  const chunks = [];
  let current = '';

  for (const piece of pieces) {
    const candidate = current ? `${current}\n\n${piece}` : piece;
    if (candidate.length <= maxChars) {
      current = candidate;
      continue;
    }

    chunks.push(current);
    const overlap = overlapLines > 0 ? current.split('\n').slice(-overlapLines).join('\n') : '';
    const withOverlap = overlap ? `${overlap}\n${piece}` : piece;
    current = withOverlap.length <= maxChars ? withOverlap : piece;
  }
  if (current) chunks.push(current);

  return chunks.map((chunkTextValue, id) => ({ id, text: chunkTextValue }));
}
