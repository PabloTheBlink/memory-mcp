import { pipeline } from '@xenova/transformers';

let extractor: any = null;

export async function getEmbedding(text: string): Promise<number[]> {
  const result = await getEmbeddings([text]);
  return result[0];
}

export async function getEmbeddings(texts: string[]): Promise<number[][]> {
  if (!extractor) {
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }

  const results: number[][] = [];
  // Process in chunks to avoid memory issues with very large batches
  const CHUNK_SIZE = 16;
  for (let i = 0; i < texts.length; i += CHUNK_SIZE) {
    const chunk = texts.slice(i, i + CHUNK_SIZE);
    const output = await extractor(chunk, { pooling: 'mean', normalize: true });
    
    // Xenova returns a single Tensor for batch, we need to split it
    const data = output.data;
    const dim = output.dims[1];
    for (let j = 0; j < chunk.length; j++) {
      results.push(Array.from(data.slice(j * dim, (j + 1) * dim)));
    }
  }
  return results;
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
  // Performance optimization: Vector Partitioning
  // For small datasets (< 200 nodes), O(N) is fine.
  // For larger datasets, we use a simple heuristic: 
  // calculate similarity only for nodes that have a positive dot product 
  // on the first 16 dimensions (coarse filtering).
  
  const shouldFilter = candidates.length > 200;
  
  return candidates
    .filter((c) => {
      if (c.embedding === null) return false;
      if (!shouldFilter) return true;
      
      // Coarse filter: quick check on a subset of dimensions
      let coarseScore = 0;
      for (let i = 0; i < 16; i++) {
        coarseScore += queryEmbedding[i] * c.embedding[i];
      }
      return coarseScore > 0; // Only keep candidates with some initial alignment
    })
    .map((c) => ({ id: c.id, similarity: cosineSimilarity(queryEmbedding, c.embedding!) }))
    .filter((r) => r.similarity >= threshold)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}
