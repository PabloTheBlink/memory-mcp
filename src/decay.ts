import {
  getAllNodes,
  getAllEdges,
  updateNodeStrength,
  updateEdgeWeight,
  deleteNode,
  deleteEdge,
  setMeta,
} from "./graph";

const NODE_DELETION_THRESHOLD = 0.05;
const EDGE_DELETION_THRESHOLD = 0.05;
const HIGH_ACCESS_THRESHOLD = 10;
const STRENGTH_BOOST = 0.1;

export interface ConsolidationStats {
  nodesProcessed: number;
  nodesDecayed: number;
  nodesDeleted: number;
  nodesStrengthened: number;
  edgesProcessed: number;
  edgesDecayed: number;
  edgesDeleted: number;
}

export function consolidate(): ConsolidationStats {
  const stats: ConsolidationStats = {
    nodesProcessed: 0,
    nodesDecayed: 0,
    nodesDeleted: 0,
    nodesStrengthened: 0,
    edgesProcessed: 0,
    edgesDecayed: 0,
    edgesDeleted: 0,
  };

  const now = Date.now();
  const nodes = getAllNodes();
  stats.nodesProcessed = nodes.length;

  for (const node of nodes) {
    const elapsedDays = (now - node.last_accessed_at) / (1000 * 60 * 60 * 24);
    
    // Human-like decay: Importance and Strength create stability.
    // Important memories (importance ~ 1.0) decay much slower.
    const stability = node.strength * (1.0 + node.importance * 5.0);
    const retention = Math.exp(-elapsedDays / Math.max(stability, 0.01));
    const newStrength = node.strength * retention;

    // "Flashbulb" memories: highly important memories are protected from deletion
    // unless they are extremely weak.
    const effectiveDeletionThreshold = node.importance > 0.8 
      ? NODE_DELETION_THRESHOLD * 0.2 
      : NODE_DELETION_THRESHOLD;

    if (newStrength < effectiveDeletionThreshold) {
      deleteNode(node.id);
      stats.nodesDeleted++;
    } else if (node.access_count >= HIGH_ACCESS_THRESHOLD) {
      // Strengthening based on access count and importance
      const reinforcement = STRENGTH_BOOST * (1.0 + node.importance);
      const boosted = Math.min(1.0, newStrength + reinforcement);
      updateNodeStrength(node.id, boosted);
      stats.nodesStrengthened++;
    } else {
      updateNodeStrength(node.id, newStrength);
      stats.nodesDecayed++;
    }
  }

  const edges = getAllEdges();
  stats.edgesProcessed = edges.length;

  for (const edge of edges) {
    const elapsedDays = (now - edge.last_reinforced_at) / (1000 * 60 * 60 * 24);
    const decayRate = 1.0 / Math.max(edge.co_occurrences * 0.5, 1);
    const retention = Math.exp(-elapsedDays * decayRate);
    const newWeight = edge.weight * retention;

    if (newWeight < EDGE_DELETION_THRESHOLD) {
      deleteEdge(edge.from_id, edge.to_id);
      stats.edgesDeleted++;
    } else {
      updateEdgeWeight(edge.from_id, edge.to_id, newWeight);
      stats.edgesDecayed++;
    }
  }

  setMeta("last_consolidation", String(now));
  return stats;
}
