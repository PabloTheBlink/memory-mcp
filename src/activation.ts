import { getNeighbors, getNodeById, touchNode, ActivatedNode } from "./graph";

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

  while (queue.length > 0) {
    const { id, activation, depth } = queue.shift()!;
    if (depth >= maxDepth) continue;

    for (const { node, edge } of getNeighbors(id)) {
      const edgeKey = [edge.from_id, edge.to_id].sort().join(":");
      edgeSet.set(edgeKey, { from_id: edge.from_id, to_id: edge.to_id, weight: edge.weight, type: edge.type });

      const spread = activation * decayFactor * edge.weight;
      if (spread < threshold) continue;

      const current = activations.get(node.id) ?? 0;
      if (spread > current) {
        activations.set(node.id, spread);
        queue.push({ id: node.id, activation: spread, depth: depth + 1 });
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
