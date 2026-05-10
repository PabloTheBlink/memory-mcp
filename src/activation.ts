import { getNeighbors, getNodeById, touchNode, fireNode, ActivatedNode } from "./graph";

export interface ActivationResult {
  nodes: ActivatedNode[];
  edges: Array<{ from_id: string; to_id: string; weight: number; type: string }>;
}

export async function spreadActivation(
  seeds: Array<{ id: string; activation: number }>,
  maxDepth: number = 3,
  decayFactor: number = 0.5,
  threshold: number = 0.1
): Promise<ActivationResult> {
  const activations = new Map<string, number>();
  const edgeSet = new Map<string, { from_id: string; to_id: string; weight: number; type: string }>();

  for (const { id, activation } of seeds) {
    activations.set(id, activation);
    touchNode(id);
  }

  const queue: Array<{ id: string; activation: number; depth: number }> = seeds.map(
    ({ id, activation }) => ({ id, activation, depth: 0 })
  );

  // Fatigue/Habituation: nodes that "fire" too much or have too many connections
  // become temporarily less receptive to prevent Hub saturation.
  const fatigue = new Map<string, number>();

  while (queue.length > 0) {
    const { id, activation, depth } = queue.shift()!;
    if (depth >= maxDepth) continue;

    const neighbors = getNeighbors(id);
    // Habituation: hubs (many neighbors) dissipate energy faster
    const degreePenalty = Math.max(0.7, 1.0 - (neighbors.length * 0.02));

    for (const { node, edge } of neighbors) {
      const edgeKey = [edge.from_id, edge.to_id].sort().join(":");
      edgeSet.set(edgeKey, { from_id: edge.from_id, to_id: edge.to_id, weight: edge.weight, type: edge.type });

      // Human-like spread: 
      // 1. Edge Type Bias: Episodic/Temporal links are strong contextual bridges.
      const typeBias = 
        edge.type === "episodic" ? 1.2 : 
        edge.type === "temporal" ? 1.1 : 
        edge.type === "causal"   ? 1.05 : 1.0;

      // 2. Importance Resonance: Important nodes attract and sustain energy.
      const targetImportance = node.importance ?? 0.5;
      const resonance = 0.9 + (targetImportance * 0.3); 
      
      // 3. Firing Fatigue: Nodes that are repeatedly triggered in one cycle 
      // lose receptivity (simulating neuronal refractory periods).
      const nodeFatigue = fatigue.get(node.id) ?? 0;
      // Persistent fatigue: if fired in the last 5 minutes, add a baseline penalty.
      const now = Date.now();
      const persistentFatigue = (now - node.last_fired_at) < 5 * 60 * 1000 ? 0.25 : 0;
      const receptivity = Math.max(0.05, 1.0 - (nodeFatigue + persistentFatigue));

      const effectiveDecay = decayFactor * resonance * typeBias * degreePenalty * receptivity; 
      const spread = activation * effectiveDecay * edge.weight;
      
      if (spread < threshold) continue;

      const current = activations.get(node.id) ?? 0;
      // Activation accumulates slightly (like neurons summing input)
      const nextActivation = Math.min(1.2, current + (spread * 0.8));

      if (nextActivation > current + 0.05) {
        activations.set(node.id, nextActivation);
        fatigue.set(node.id, nodeFatigue + 0.2); // Increase session fatigue
        fireNode(node.id); // Mark as fired persistently
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
