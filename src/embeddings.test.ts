import { describe, it, expect, vi, beforeEach } from 'vitest';
import { cosineSimilarity, findSimilar } from './embeddings';

describe('embeddings', () => {
  describe('cosineSimilarity', () => {
    it('should return 1 for identical vectors', () => {
      const v = [1, 0, 0];
      expect(cosineSimilarity(v, v)).toBeCloseTo(1);
    });

    it('should return 0 for orthogonal vectors', () => {
      const v1 = [1, 0, 0];
      const v2 = [0, 1, 0];
      expect(cosineSimilarity(v1, v2)).toBe(0);
    });

    it('should return -1 for opposite vectors', () => {
      const v1 = [1, 0, 0];
      const v2 = [-1, 0, 0];
      expect(cosineSimilarity(v1, v2)).toBeCloseTo(-1);
    });

    it('should return 0 for zero vectors', () => {
      expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
    });
  });

  describe('findSimilar', () => {
    const query = [1, 0, 0];
    const candidates = [
      { id: '1', embedding: [1, 0, 0] },     // similarity 1
      { id: '2', embedding: [0.8, 0.6, 0] }, // similarity 0.8
      { id: '3', embedding: [0, 1, 0] },     // similarity 0
      { id: '4', embedding: null },           // should be ignored
    ];

    it('should return similar items above threshold', () => {
      const results = findSimilar(query, candidates, 0.5);
      expect(results).toHaveLength(2);
      expect(results[0].id).toBe('1');
      expect(results[1].id).toBe('2');
    });

    it('should sort by similarity descending', () => {
      const results = findSimilar(query, candidates, 0.1);
      expect(results[0].similarity).toBeGreaterThan(results[1].similarity);
    });

    it('should respect the limit', () => {
      const results = findSimilar(query, candidates, 0.1, 1);
      expect(results).toHaveLength(1);
    });
  });
});
