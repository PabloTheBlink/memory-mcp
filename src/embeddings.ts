import { pipeline } from '@xenova/transformers';

let extractor: any = null;

export async function getEmbedding(text: string): Promise<number[]> {
  if (!extractor) {
    // Initializing the model (Xenova/all-MiniLM-L6-v2 is lightweight and effective)
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }

  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function findSimilar(
  queryEmbedding: number[],
  candidates: Array<{ id: string; embedding: number[] | null }>,
  threshold: number = 0.75,
  limit: number = 20
): Array<{ id: string; similarity: number }> {
  return candidates
    .filter((c) => c.embedding !== null)
    .map((c) => ({ id: c.id, similarity: cosineSimilarity(queryEmbedding, c.embedding!) }))
    .filter((r) => r.similarity >= threshold)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}
