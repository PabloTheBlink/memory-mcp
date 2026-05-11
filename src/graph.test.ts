import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setDbPath, closeDb, findOrCreateNode, getNodeById, getAllNodes, upsertEdge, getNeighbors, deleteNode, getStats } from './graph';
import fs from 'fs';
import path from 'path';

describe('graph', () => {
  beforeEach(() => {
    // Use in-memory database for each test
    setDbPath(':memory:');
  });

  afterEach(() => {
    closeDb();
  });

  describe('findOrCreateNode', () => {
    it('should create a new node if it does not exist', () => {
      const node = findOrCreateNode('test-label');
      expect(node.label).toBe('test-label');
      expect(node.id).toBeDefined();
      
      const nodes = getAllNodes();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].label).toBe('test-label');
    });

    it('should return existing node if label matches', () => {
      const node1 = findOrCreateNode('test-label');
      const node2 = findOrCreateNode('test-label');
      
      expect(node1.id).toBe(node2.id);
      expect(getAllNodes()).toHaveLength(1);
    });

    it('should store metadata', () => {
      const metadata = { foo: 'bar', count: 42 };
      const node = findOrCreateNode('meta-node', null, 0.5, metadata);
      
      expect(node.metadata).toEqual(metadata);
      
      const retrieved = getNodeById(node.id);
      expect(retrieved?.metadata).toEqual(metadata);
    });
  });

  describe('edges', () => {
    it('should create and retrieve edges', () => {
      const n1 = findOrCreateNode('n1');
      const n2 = findOrCreateNode('n2');
      
      const edge = upsertEdge(n1.id, n2.id, 'semantic');
      expect(edge.from_id).toBe(n1.id < n2.id ? n1.id : n2.id);
      expect(edge.to_id).toBe(n1.id < n2.id ? n2.id : n1.id);
      
      const neighbors = getNeighbors(n1.id);
      expect(neighbors).toHaveLength(1);
      expect(neighbors[0].node.id).toBe(n2.id);
    });

    it('should boost weight on reinforcement', () => {
      const n1 = findOrCreateNode('n1');
      const n2 = findOrCreateNode('n2');
      
      const e1 = upsertEdge(n1.id, n2.id, 'semantic', 0.1);
      const initialWeight = e1.weight;
      
      const e2 = upsertEdge(n1.id, n2.id, 'semantic', 0.2);
      expect(e2.weight).toBeCloseTo(initialWeight + 0.2);
      expect(e2.co_occurrences).toBe(2);
    });
  });

  describe('deletion', () => {
    it('should delete a node and its edges', () => {
      const n1 = findOrCreateNode('n1');
      const n2 = findOrCreateNode('n2');
      upsertEdge(n1.id, n2.id, 'semantic');
      
      deleteNode(n1.id);
      
      expect(getNodeById(n1.id)).toBeNull();
      expect(getNeighbors(n2.id)).toHaveLength(0);
      expect(getStats().nodeCount).toBe(1);
    });
  });
});
