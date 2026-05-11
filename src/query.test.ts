import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setDbPath, closeDb, findOrCreateNode, upsertEdge, updateNodeEmbedding } from './graph';
import { queryMemory } from './query';
import * as embeddings from './embeddings';

// Mock embeddings to avoid Ollama calls
vi.mock('./embeddings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./embeddings')>();
  return {
    ...actual,
    getEmbedding: vi.fn().mockResolvedValue(new Array(384).fill(0.1)),
    findSimilar: vi.fn(),
  };
});

describe('query', () => {
  beforeEach(() => {
    setDbPath(':memory:');
    vi.clearAllMocks();
  });

  afterEach(() => {
    closeDb();
  });

  it('should return null if prompt is too short', async () => {
    const result = await queryMemory('hi');
    expect(result).toBeNull();
  });

  it('should return context when similar nodes are found', async () => {
    const n1 = findOrCreateNode('Relevant Memory');
    updateNodeEmbedding(n1.id, new Array(384).fill(0.1));
    
    // Mock findSimilar to return our node
    vi.mocked(embeddings.findSimilar).mockReturnValue([
      { id: n1.id, label: n1.label, similarity: 0.9 }
    ]);

    const result = await queryMemory('Tell me about my relevant memory');
    
    expect(result).toContain('Relevant Memory');
    expect(result).toContain('[Memory Context');
  });

  it('should include active rules in context', async () => {
    const rule = findOrCreateNode('rule:Use TypeScript');
    updateNodeEmbedding(rule.id, new Array(384).fill(0.1));
    
    const n1 = findOrCreateNode('Other Memory');
    updateNodeEmbedding(n1.id, new Array(384).fill(0.1));

    // Mock findSimilar to return at least one memory so we don't bail early
    vi.mocked(embeddings.findSimilar).mockReturnValue([
      { id: n1.id, label: n1.label, similarity: 0.9 }
    ]);
    
    const result = await queryMemory('How should I code?');
    
    expect(result).not.toBeNull();
    expect(result).toContain('Use TypeScript');
    expect(result).toContain('[Active Rules & Preferences]');
  });
});
