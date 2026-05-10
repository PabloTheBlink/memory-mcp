import { getNeighbors, getNodeById, touchNode, fireNode, ActivatedNode } from "./graph";

export interface ActivationResult {
  nodes: ActivatedNode[];
  edges: Array<{ from_id: string; to_id: string; weight: number; type: string }>;
}

export async function spreadActivation(
  seeds: Array<{ id: string; activation: number }>,
  maxDepth: number = 3,
  decayFactor: number = 0.5,
  threshold: number = 0.1,
  activeContextNodeId?: string
): Promise<ActivationResult> {
  const activations = new Map<string, number>();
  const edgeSet = new Map<string, { from_id: string; to_id: string; weight: number; type: string }>();

  // Pre-fetch active context neighbors for efficient inhibition
  const contextNeighbors = new Set<string>();
  if (activeContextNodeId) {
    contextNeighbors.add(activeContextNodeId);
    getNeighbors(activeContextNodeId).forEach(n => contextNeighbors.add(n.node.id));
  }

  for (const { id, activation } of seeds) {
    activations.set(id, activation);
    touchNode(id);
  }

  const queue: Array<{ id: string; activation: number; depth: number }> = seeds.map(
    ({ id, activation }) => ({ id, activation, depth: 0 })
  );

  const fatigue = new Map<string, number>();

  while (queue.length > 0) {
    const { id, activation, depth } = queue.shift()!;
    if (depth >= maxDepth) continue;

    const neighbors = getNeighbors(id);
    const degreePenalty = Math.max(0.7, 1.0 - (neighbors.length * 0.02));

    const currentActivation = activations.get(id) ?? 0;
    const inhibitionRadius = currentActivation > 0.8 ? 0.05 : 0;

    for (const { node, edge } of neighbors) {
      const edgeKey = [edge.from_id, edge.to_id].sort().join(":");
      edgeSet.set(edgeKey, { from_id: edge.from_id, to_id: edge.to_id, weight: edge.weight, type: edge.type });

      // Human-like spread logic: favor semantic and conceptual links
      const typeBias = 
        edge.type === "episodic" ? 1.25 : 
        edge.type === "semantic" ? 1.20 : 
        edge.type === "abstraction" ? 1.15 :
        edge.type === "temporal" ? 1.10 : 
        edge.type === "causal"   ? 1.05 : 1.0;

      const targetImportance = node.importance ?? 0.5;
      const resonance = 0.9 + (targetImportance * 0.4); 
      
      const nodeFatigue = fatigue.get(node.id) ?? 0;
      const now = Date.now();
      const persistentFatigue = (now - node.last_fired_at) < 5 * 60 * 1000 ? 0.3 : 0;
      const receptivity = Math.max(0.02, 1.0 - (nodeFatigue + persistentFatigue));

      // Contextual Inhibition: if we have an active context, suppress nodes not associated with it.
      // This sharpens focus and prevents "cross-talk" between unrelated memory clusters.
      let contextInhibition = 0;
      if (activeContextNodeId && !contextNeighbors.has(node.id)) {
        // Label-based check for "global" nodes that shouldn't be inhibited (e.g. personal info)
        const isGlobal = node.label.toLowerCase().includes("pablo") || node.label.includes("streak");
        if (!isGlobal) {
          contextInhibition = 0.4; // Strong penalty for off-context memories
        }
      }

      const effectiveDecay = decayFactor * resonance * typeBias * degreePenalty * receptivity; 
      const spread = (activation - (inhibitionRadius + contextInhibition)) * effectiveDecay * edge.weight;
      
      if (spread < threshold) continue;

      const current = activations.get(node.id) ?? 0;
      const nextActivation = Math.min(1.3, current + (spread * 0.85));

      if (nextActivation > current + 0.03) {
        activations.set(node.id, nextActivation);
        fatigue.set(node.id, nodeFatigue + 0.25); 
        fireNode(node.id); 
        queue.push({ id: node.id, activation: nextActivation, depth: depth + 1 });
      }
    }
  }

  const nodes: ActivatedNode[] = [];
  for (const [nodeId, activation] of activations) {
    const n = getNodeById(nodeId);
    if (n) nodes.push({ ...n, activation });
  }

  return {
    nodes: nodes.sort((a, b) => b.activation - a.activation),
    edges: Array.from(edgeSet.values()),
  };
}
