import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setDbPath, closeDb, findOrCreateNode, getNodeById, getAllNodes, upsertEdge, getNeighbors, deleteNode, getStats } from './graph';
import fs from 'fs';
import path from 'path';

describe('graph', () => {
  beforeEach(async () => {
    // Use in-memory database for each test
    setDbPath(':memory:');
  });

  afterEach(() => {
    closeDb();
  });

  describe('findOrCreateNode', () => {
    it('should create a new node if it does not exist', async () => {
      const node = await findOrCreateNode('test-label');
      expect(node.label).toBe('test-label');
      expect(node.id).toBeDefined();
      
      const nodes = await getAllNodes();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].label).toBe('test-label');
    });

    it('should return existing node if label matches', async () => {
      const node1 = await findOrCreateNode('test-label');
      const node2 = await findOrCreateNode('test-label');
      
      expect(node1.id).toBe(node2.id);
      expect(await getAllNodes()).toHaveLength(1);
    });

    it('should store metadata', async () => {
      const metadata = { foo: 'bar', count: 42 };
      const node = await findOrCreateNode('meta-node', null, 0.5, metadata);
      
      expect(node.metadata).toEqual(metadata);
      
      const retrieved = await getNodeById(node.id);
      expect(retrieved?.metadata).toEqual(metadata);
    });
  });

  describe('edges', () => {
    it('should create and retrieve edges', async () => {
      const n1 = await findOrCreateNode('n1');
      const n2 = await findOrCreateNode('n2');
      
      const edge = await upsertEdge(n1.id, n2.id, 'semantic');
      expect(edge.from_id).toBe(n1.id < n2.id ? n1.id : n2.id);
      expect(edge.to_id).toBe(n1.id < n2.id ? n2.id : n1.id);
      
      const neighbors = await getNeighbors(n1.id);
      expect(neighbors).toHaveLength(1);
      expect(neighbors[0].node.id).toBe(n2.id);
    });

    it('should boost weight on reinforcement', async () => {
      const n1 = await findOrCreateNode('n1');
      const n2 = await findOrCreateNode('n2');
      
      const e1 = await upsertEdge(n1.id, n2.id, 'semantic', 0.1);
      const initialWeight = e1.weight;
      
      const e2 = await upsertEdge(n1.id, n2.id, 'semantic', 0.2);
      expect(e2.weight).toBeCloseTo(initialWeight + 0.2);
      expect(e2.co_occurrences).toBe(2);
    });
  });

  describe('deletion', () => {
    it('should delete a node and its edges', async () => {
      const n1 = await findOrCreateNode('n1');
      const n2 = await findOrCreateNode('n2');
      await upsertEdge(n1.id, n2.id, 'semantic');
      
      await deleteNode(n1.id);
      
      expect(await getNodeById(n1.id)).toBeNull();
      expect(await getNeighbors(n2.id)).toHaveLength(0);
      expect((await getStats()).nodeCount).toBe(1);
    });
  });
});
