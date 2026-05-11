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

  it('should decrease strength over time', async () => {
    const now = Date.now();
    vi.setSystemTime(now);
    
    // Create node with high importance to slow down decay
    const node = await findOrCreateNode('decaying-node', null, 0.9);
    const initialStrength = node.strength;
    
    // Move time forward by 1 day
    vi.setSystemTime(now + 1 * 24 * 60 * 60 * 1000);
    
    const stats = await consolidate();
    expect(stats.nodesDecayed).toBe(1);
    
    const decayedNode = await getNodeById(node.id);
    expect(decayedNode!.strength).toBeLessThan(initialStrength);
  });

  it('should delete nodes that fall below threshold', async () => {
    const now = Date.now();
    vi.setSystemTime(now);
    
    const node = await findOrCreateNode('forgotten-node', null, 0.1);
    
    // Move time forward by 100 days
    vi.setSystemTime(now + 100 * 24 * 60 * 60 * 1000);
    
    const stats = await consolidate();
    expect(stats.nodesDeleted).toBe(1);
    expect(await getNodeById(node.id)).toBeNull();
  });

  it('should handle edge decay', async () => {
    const now = Date.now();
    vi.setSystemTime(now);
    
    const n1 = await findOrCreateNode('n1');
    const n2 = await findOrCreateNode('n2');
    const edge = await upsertEdge(n1.id, n2.id, 'semantic');
    const initialWeight = edge.weight;
    
    // Move time forward by 1 hour (very small decay)
    vi.setSystemTime(now + 1 * 60 * 60 * 1000);
    
    const stats = await consolidate();
    expect(stats.edgesDecayed).toBe(1);
    
    const neighbors = await getNeighbors(n1.id);
    expect(neighbors[0].edge.weight).toBeLessThan(initialWeight);
  });
});
