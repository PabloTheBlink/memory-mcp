import { pipeline, env } from '@xenova/transformers';

// Disable progress bars and logging to stdout, which would break the MCP protocol
(env as any).showProgressBar = false;
(env as any).allowRemoteModels = true; 
(env as any).allowLocalModels = true;

let extractor: any = null;
const embeddingCache = new Map<string, number[]>();

export async function getEmbedding(text: string): Promise<number[]> {
  if (embeddingCache.has(text)) return embeddingCache.get(text)!;
  const result = await getEmbeddings([text]);
  embeddingCache.set(text, result[0]);
  return result[0];
}

export async function getEmbeddings(texts: string[]): Promise<number[][]> {
  if (!extractor) {
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }

  const results: number[][] = [];
  const toFetch: string[] = [];
  const fetchIndices: number[] = [];

  for (let i = 0; i < texts.length; i++) {
    if (embeddingCache.has(texts[i])) {
      results[i] = embeddingCache.get(texts[i])!;
    } else {
      toFetch.push(texts[i]);
      fetchIndices.push(i);
    }
  }

  if (toFetch.length > 0) {
    const CHUNK_SIZE = 16;
    for (let i = 0; i < toFetch.length; i += CHUNK_SIZE) {
      const chunk = toFetch.slice(i, i + CHUNK_SIZE);
      const output = await extractor(chunk, { pooling: 'mean', normalize: true });
      
      const data = output.data;
      const dim = output.dims[1];
      for (let j = 0; j < chunk.length; j++) {
        const emb = Array.from(data.slice(j * dim, (j + 1) * dim)) as number[];
        const originalIndex = fetchIndices[i + j];
        results[originalIndex] = emb;
        embeddingCache.set(toFetch[i + j], emb);
      }
    }
  }

  return results;
}

export function cosineSimilarity(a: number[] | Float32Array, b: number[] | Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) {
    const ai = a[i];
    const bi = b[i];
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
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
