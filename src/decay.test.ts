import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { setDbPath, closeDb, findOrCreateNode, getNodeById, getAllNodes, upsertEdge, getNeighbors } from './graph';
import { consolidate } from './decay';

describe('decay', () => {
  beforeEach(() => {
    setDbPath(':memory:');
    vi.useFakeTimers();
  });

  afterEach(() => {
    closeDb();
    vi.useRealTimers();
  });

  it('should decrease strength over time', () => {
    const now = Date.now();
    vi.setSystemTime(now);
    
    // Create node with high importance to slow down decay
    const node = findOrCreateNode('decaying-node', null, 0.9);
    const initialStrength = node.strength;
    
    // Move time forward by 1 day
    vi.setSystemTime(now + 1 * 24 * 60 * 60 * 1000);
    
    const stats = consolidate();
    expect(stats.nodesDecayed).toBe(1);
    
    const decayedNode = getNodeById(node.id);
    expect(decayedNode!.strength).toBeLessThan(initialStrength);
  });

  it('should delete nodes that fall below threshold', () => {
    const now = Date.now();
    vi.setSystemTime(now);
    
    const node = findOrCreateNode('forgotten-node', null, 0.1);
    
    // Move time forward by 100 days
    vi.setSystemTime(now + 100 * 24 * 60 * 60 * 1000);
    
    const stats = consolidate();
    expect(stats.nodesDeleted).toBe(1);
    expect(getNodeById(node.id)).toBeNull();
  });

  it('should handle edge decay', () => {
    const now = Date.now();
    vi.setSystemTime(now);
    
    const n1 = findOrCreateNode('n1');
    const n2 = findOrCreateNode('n2');
    const edge = upsertEdge(n1.id, n2.id, 'semantic');
    const initialWeight = edge.weight;
    
    // Move time forward by 1 hour (very small decay)
    vi.setSystemTime(now + 1 * 60 * 60 * 1000);
    
    const stats = consolidate();
    expect(stats.edgesDecayed).toBe(1);
    
    const neighbors = getNeighbors(n1.id);
    expect(neighbors[0].edge.weight).toBeLessThan(initialWeight);
  });
});
