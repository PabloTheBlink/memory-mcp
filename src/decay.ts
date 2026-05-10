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
    const retention = Math.exp(-elapsedDays / Math.max(node.strength, 0.01));
    const newStrength = node.strength * retention;

    if (newStrength < NODE_DELETION_THRESHOLD) {
      deleteNode(node.id);
      stats.nodesDeleted++;
    } else if (node.access_count >= HIGH_ACCESS_THRESHOLD) {
      const boosted = Math.min(1.0, newStrength + STRENGTH_BOOST);
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
