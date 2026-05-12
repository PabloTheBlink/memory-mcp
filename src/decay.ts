import {
  getAllNodes,
  getAllEdges,
  updateNodeStrength,
  updateEdgeWeight,
  deleteNode,
  deleteEdge,
  setMeta,
} from "./graph";
import { cosineSimilarity } from "./embeddings";

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

export async function consolidate(): Promise<ConsolidationStats> {
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
  const nodes = await getAllNodes();
  stats.nodesProcessed = nodes.length;

  // ── Step 1.5: Interference (Inhibition) ─────────────────────────────
  const interferencePenalties = new Map<string, number>();
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      if (!a.embedding || !b.embedding) continue;
      
      const sim = cosineSimilarity(a.embedding, b.embedding);
      if (sim > 0.85) {
        const aRecent = a.last_accessed_at;
        const bRecent = b.last_accessed_at;
        if (Math.abs(aRecent - bRecent) > 1000 * 60 * 60) {
          const [older, newer] = aRecent < bRecent ? [a, b] : [b, a];
          const penalty = 0.05 * (sim - 0.8) * 5; 
          interferencePenalties.set(older.id, (interferencePenalties.get(older.id) ?? 0) + penalty);
        }
      }
    }
  }

  for (const node of nodes) {
    const elapsedDays = (now - node.last_accessed_at) / (1000 * 60 * 60 * 24);
    const baseStability = Math.max(0.1, node.importance * 2.0);
    const spacingMultiplier = Math.pow(1.8, Math.min(node.access_count, 15));
    const stability = baseStability * spacingMultiplier;
    
    const retention = Math.exp(-elapsedDays / stability);
    let newStrength = node.strength * retention;

    const penalty = interferencePenalties.get(node.id) ?? 0;
    newStrength = Math.max(0, newStrength - penalty);

    const effectiveDeletionThreshold = node.importance > 0.8 
      ? NODE_DELETION_THRESHOLD * 0.2 
      : NODE_DELETION_THRESHOLD;

    if (newStrength < effectiveDeletionThreshold) {
      await deleteNode(node.id);
      stats.nodesDeleted++;
    } else {
      if (elapsedDays < 1.0 && node.access_count > 0 && penalty === 0) {
        const retrievalEffort = 1.0 + (1.0 - retention);
        const reinforcement = STRENGTH_BOOST * (1.0 + node.importance) * retrievalEffort;
        newStrength = Math.min(1.0, newStrength + reinforcement);
        await updateNodeStrength(node.id, newStrength);
        stats.nodesStrengthened++;
      } else {
        await updateNodeStrength(node.id, newStrength);
        stats.nodesDecayed++;
      }
    }
  }

  const edges = await getAllEdges();
  stats.edgesProcessed = edges.length;

  for (const edge of edges) {
    const elapsedDays = (now - edge.last_reinforced_at) / (1000 * 60 * 60 * 24);
    const decayRate = 1.0 / Math.max(edge.co_occurrences * 0.5, 1);
    const retention = Math.exp(-elapsedDays * decayRate);
    const newWeight = edge.weight * retention;

    if (newWeight < EDGE_DELETION_THRESHOLD) {
      await deleteEdge(edge.from_id, edge.to_id);
      stats.edgesDeleted++;
    } else {
      await updateEdgeWeight(edge.from_id, edge.to_id, newWeight);
      stats.edgesDecayed++;
    }
  }

  await setMeta("last_consolidation", String(now));
  return stats;
}
