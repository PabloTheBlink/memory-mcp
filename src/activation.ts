import { getNeighbors, getNeighborsBatch, getNodeById, touchNode, fireNode, suggestNode, ActivatedNode } from "./graph";

export interface ActivationResult {
  nodes: ActivatedNode[];
  edges: Array<{ from_id: string; to_id: string; weight: number; type: string }>;
}

export async function spreadActivation(
  seeds: Array<{ id: string; activation: number }>,
  maxDepth: number = 3,
  decayFactor: number = 0.5,
  threshold: number = 0.1,
  activeContextNodeId?: string,
  isProactive: boolean = false
): Promise<ActivationResult> {
  const activations = new Map<string, number>();
  const edgeSet = new Map<string, { from_id: string; to_id: string; weight: number; type: string }>();

  // Pre-fetch active context and its hierarchy (parents) for efficient focus
  const contextNeighbors = new Set<string>();
  if (activeContextNodeId) {
    const queue = [activeContextNodeId];
    const visited = new Set<string>();
    while (queue.length > 0) {
      const cid = queue.shift()!;
      if (visited.has(cid)) continue;
      visited.add(cid);
      contextNeighbors.add(cid);
      
      const neighbors = await getNeighbors(cid);
      for (const n of neighbors) {
        contextNeighbors.add(n.node.id);
        // If it's a parent (abstraction link where we are the child), add to queue to find grandparents
        if (n.edge.type === "abstraction" && n.edge.to_id === cid) {
           queue.push(n.node.id);
        }
      }
    }
  }

  for (const { id, activation } of seeds) {
    activations.set(id, activation);
    await touchNode(id);
  }

  let currentLevel: Array<{ id: string; activation: number }> = [...seeds];
  const fatigue = new Map<string, number>();
  const visitedNodes = new Set<string>();

  for (let depth = 0; depth < maxDepth; depth++) {
    if (currentLevel.length === 0) break;

    const nodeIds = currentLevel.map(n => n.id);
    const neighborsMap = await getNeighborsBatch(nodeIds);
    const nextLevel: Array<{ id: string; activation: number }> = [];

    for (const { id, activation } of currentLevel) {
      if (visitedNodes.has(id)) continue;
      // visitedNodes.add(id); // Optional: allow re-activation if new signal is stronger? 
      // In this version, we'll allow re-activation to support complex resonance.

      const neighbors = neighborsMap.get(id) || [];
      const degreePenalty = Math.max(0.7, 1.0 - (neighbors.length * 0.02));
      const inhibitionRadius = activation > 0.8 ? 0.05 : 0;

      for (const { node, edge } of neighbors) {
        const edgeKey = [edge.from_id, edge.to_id].sort().join(":");
        edgeSet.set(edgeKey, { from_id: edge.from_id, to_id: edge.to_id, weight: edge.weight, type: edge.type });

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

        let contextInhibition = 0;
        if (activeContextNodeId && !contextNeighbors.has(node.id)) {
          const isGlobal = node.label.toLowerCase().includes("pablo") || node.label.includes("streak");
          if (!isGlobal) contextInhibition = 0.4;
        }

        const effectiveDecay = decayFactor * resonance * typeBias * degreePenalty * receptivity; 
        const spread = (activation - (inhibitionRadius + contextInhibition)) * effectiveDecay * edge.weight;
        
        if (spread < threshold) continue;

        const current = activations.get(node.id) ?? 0;
        const nextActivation = Math.min(1.0, current + (spread * 0.85));

        if (nextActivation > current + threshold) {
          activations.set(node.id, nextActivation);
          fatigue.set(node.id, nodeFatigue + 0.25); 
          if (isProactive) {
            await suggestNode(node.id);
          } else {
            await fireNode(node.id);
          }
          nextLevel.push({ id: node.id, activation: nextActivation });
        }
      }
    }
    currentLevel = nextLevel;
  }

  // Final batch fetch of all activated nodes to return full objects
  const finalIds = Array.from(activations.keys());
  const nodes: ActivatedNode[] = [];
  
  // We don't have a getNodesBatch but we can use getAllNodes and filter if count is large, 
  // or just multiple getNodeById calls. For now, multiple is fine as it's the final step.
  for (const nodeId of finalIds) {
    const n = await getNodeById(nodeId);
    if (n) nodes.push({ ...n, activation: activations.get(nodeId)! });
  }

  return {
    nodes: nodes.sort((a, b) => b.activation - a.activation),
    edges: Array.from(edgeSet.values()),
  };
}
