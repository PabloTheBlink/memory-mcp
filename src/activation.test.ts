import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setDbPath, closeDb, findOrCreateNode, upsertEdge } from './graph';
import { spreadActivation } from './activation';

describe('activation', () => {
  beforeEach(() => {
    setDbPath(':memory:');
    vi.useFakeTimers();
  });

  afterEach(() => {
    closeDb();
    vi.useRealTimers();
  });

  it('should spread activation to neighbors', async () => {
    const n1 = findOrCreateNode('n1');
    const n2 = findOrCreateNode('n2');
    upsertEdge(n1.id, n2.id, 'semantic');

    const result = await spreadActivation([{ id: n1.id, activation: 1.0 }]);
    
    expect(result.nodes).toHaveLength(2);
    const n2Result = result.nodes.find(n => n.id === n2.id);
    expect(n2Result).toBeDefined();
    expect(n2Result!.activation).toBeGreaterThan(0);
  });

  it('should respect context inhibition', async () => {
    const ctx = findOrCreateNode('context-node');
    const n1 = findOrCreateNode('in-context');
    const n2 = findOrCreateNode('out-of-context');
    const seed = findOrCreateNode('seed');

    // seed is linked to both
    upsertEdge(seed.id, n1.id, 'semantic');
    upsertEdge(seed.id, n2.id, 'semantic');

    // n1 is linked to context
    upsertEdge(ctx.id, n1.id, 'episodic');

    // Spread with context
    const result = await spreadActivation(
      [{ id: seed.id, activation: 1.0 }],
      3, 0.5, 0.01, ctx.id
    );

    const n1Res = result.nodes.find(n => n.id === n1.id);
    const n2Res = result.nodes.find(n => n.id === n2.id);

    expect(n1Res).toBeDefined();
    expect(n2Res).toBeDefined();
    
    // n2 should have lower activation than n1 because it's inhibited by context mismatch
    // (Note: there might be some noise due to weights, but here they are same)
    expect(n1Res!.activation).toBeGreaterThan(n2Res!.activation);
  });

  it('should respect maxDepth', async () => {
    const n1 = findOrCreateNode('n1');
    const n2 = findOrCreateNode('n2');
    const n3 = findOrCreateNode('n3');
    const n4 = findOrCreateNode('n4');

    upsertEdge(n1.id, n2.id, 'semantic');
    upsertEdge(n2.id, n3.id, 'semantic');
    upsertEdge(n3.id, n4.id, 'semantic');

    const result = await spreadActivation([{ id: n1.id, activation: 1.0 }], 2, 0.5, 0.01);
    
    // depth 0: n1
    // depth 1: n2
    // depth 2: n3
    // depth 3: n4 (should not be reached)
    expect(result.nodes.map(n => n.label)).toContain('n1');
    expect(result.nodes.map(n => n.label)).toContain('n2');
    expect(result.nodes.map(n => n.label)).toContain('n3');
    expect(result.nodes.map(n => n.label)).not.toContain('n4');
  });
});
