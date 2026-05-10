import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import {
  findOrCreateNode,
  getAllNodes,
  getAllEdges,
  getTopNodes,
  getStats,
  updateNodeEmbedding,
  upsertEdge,
  touchNode,
  setMeta,
} from "./graph";
import { getEmbedding, findSimilar } from "./embeddings";
import { spreadActivation } from "./activation";
import { consolidate } from "./decay";
import { runMaintenance } from "./maintenance";
import {
  getActiveContext,
  ensureContextNode,
  bindToContext,
  contextLabel,
  isContextNode,
  detectContext,
} from "./context";

const server = new Server(
  { name: "memory-mcp", version: "1.1.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "memory_activate",
      description:
        "Activate a concept in memory. Finds or creates the node, runs spreading activation seeded from both the concept and the active context, returns the activated subgraph. Concepts learned in a context become easier to recall within that context.",
      inputSchema: {
        type: "object",
        properties: {
          concept: { type: "string", description: "The concept to activate" },
        },
        required: ["concept"],
      },
    },
    {
      name: "memory_associate",
      description: "Create or reinforce an association between two concepts.",
      inputSchema: {
        type: "object",
        properties: {
          concept_a: { type: "string" },
          concept_b: { type: "string" },
          type: {
            type: "string",
            enum: ["causal", "temporal", "semantic", "episodic"],
            default: "semantic",
          },
          strength: { type: "number", minimum: 0, maximum: 1 },
        },
        required: ["concept_a", "concept_b"],
      },
    },
    {
      name: "memory_recall",
      description:
        "Recall memories relevant to a query. Uses semantic similarity + spreading activation biased toward the active context. Memories from any context can surface if strongly associated.",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string" },
          top_k: { type: "number", default: 10 },
        },
        required: ["query"],
      },
    },
    {
      name: "memory_consolidate",
      description:
        "Apply Ebbinghaus decay to all nodes and edges. Prune weak memories, strengthen frequently accessed ones.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "memory_status",
      description: "Return memory system statistics including active context.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "memory_maintenance",
      description:
        "Full maintenance pass: Ebbinghaus decay, semantic linking (connect nodes that are similar but unconnected), auto-merge near-duplicates, orphan pruning. Safe to call anytime — skips if run less than 1 hour ago unless force=true.",
      inputSchema: {
        type: "object",
        properties: {
          force: { type: "boolean", default: false, description: "Run even if maintenance ran recently" },
        },
      },
    },
    {
      name: "memory_set_context",
      description:
        "Set the active memory context. Context is a node hub — activating it biases recall toward memories formed in that context, just like how physical environment cues memory in humans. Use 'user' for personal info, 'project:<name>' for project-specific knowledge.",
      inputSchema: {
        type: "object",
        properties: {
          context: {
            type: "string",
            description:
              "Context name. Examples: 'user', 'project:devetty-platform', 'project:my-app'. Leave empty to auto-detect from git.",
          },
        },
        required: [],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "memory_activate": {
        const concept = (args as any).concept as string;
        const activeContext = getActiveContext();

        let node = findOrCreateNode(concept);
        if (!node.embedding) {
          const emb = await getEmbedding(concept);
          updateNodeEmbedding(node.id, emb);
          node = { ...node, embedding: emb };
        }

        // Find semantically similar nodes and wire them
        const allNodes = getAllNodes();
        const similar = findSimilar(node.embedding!, allNodes, 0.75, 20).filter((s) => s.id !== node.id);
        for (const s of similar.slice(0, 5)) {
          upsertEdge(node.id, s.id, "semantic", s.similarity * 0.1);
        }

        // Bind to active context (episodic encoding — this happened here)
        const contextNodeId = await ensureContextNode(activeContext);
        bindToContext(node.id, contextNodeId);

        touchNode(node.id);

        // Spread from concept (full activation) + context hub (partial, like environmental priming)
        const result = await spreadActivation(
          [
            { id: node.id, activation: 1.0 },
            { id: contextNodeId, activation: 0.3 },
          ],
          3,
          0.5,
          0.1
        );

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  activated_concept: concept,
                  active_context: activeContext,
                  node_id: node.id,
                  similar_concepts: similar.slice(0, 5).map((s) => {
                    const n = allNodes.find((nn) => nn.id === s.id);
                    return { label: n?.label ?? s.id, similarity: s.similarity };
                  }),
                  activated_subgraph: {
                    nodes: result.nodes
                      .filter((n) => !isContextNode(n.label))
                      .map((n) => ({
                        id: n.id,
                        label: n.label,
                        activation: n.activation,
                        strength: n.strength,
                        access_count: n.access_count,
                      })),
                    context_nodes: result.nodes
                      .filter((n) => isContextNode(n.label))
                      .map((n) => ({ label: n.label, activation: n.activation })),
                    edges: result.edges,
                  },
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "memory_associate": {
        const { concept_a, concept_b, type = "semantic", strength } = args as any;
        const activeContext = getActiveContext();

        let nodeA = findOrCreateNode(concept_a);
        let nodeB = findOrCreateNode(concept_b);

        if (!nodeA.embedding) {
          const emb = await getEmbedding(concept_a);
          updateNodeEmbedding(nodeA.id, emb);
        }
        if (!nodeB.embedding) {
          const emb = await getEmbedding(concept_b);
          updateNodeEmbedding(nodeB.id, emb);
        }

        const boost = strength !== undefined ? strength * 0.2 : 0.1;
        const edge = upsertEdge(nodeA.id, nodeB.id, type, boost);

        // Bind both to context
        const contextNodeId = await ensureContextNode(activeContext);
        bindToContext(nodeA.id, contextNodeId);
        bindToContext(nodeB.id, contextNodeId);

        touchNode(nodeA.id);
        touchNode(nodeB.id);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  edge: {
                    from: concept_a,
                    to: concept_b,
                    weight: edge.weight,
                    type: edge.type,
                    co_occurrences: edge.co_occurrences,
                  },
                  context: activeContext,
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "memory_recall": {
        const { query, top_k = 10 } = args as any;
        const activeContext = getActiveContext();

        const queryEmbedding = await getEmbedding(query);
        const allNodes = getAllNodes();
        // 0.35 threshold handles cross-language queries (e.g. Spanish query → English nodes)
        const similar = findSimilar(queryEmbedding, allNodes, 0.35, top_k * 2);

        // Context hub gets medium activation — it primes context-relevant memories
        // without blocking cross-context ones (like remembering work things at home)
        const contextNodeId = await ensureContextNode(activeContext);
        const seeds: Array<{ id: string; activation: number }> = [
          { id: contextNodeId, activation: 0.4 },
          ...similar.slice(0, 5).map((s) => ({ id: s.id, activation: 1.0 })),
        ];

        if (seeds.length === 1) {
          // Only context seed — no semantic matches at all
          return {
            content: [{ type: "text", text: JSON.stringify({ query, context: activeContext, results: [] }) }],
          };
        }

        for (const { id } of seeds) touchNode(id);
        const result = await spreadActivation(seeds, 3, 0.5, 0.05);

        const similarityMap = new Map(similar.map((s) => [s.id, s.similarity]));

        const ranked = result.nodes
          .filter((n) => !isContextNode(n.label))
          .map((n) => ({
            id: n.id,
            label: n.label,
            // Relevance blends semantic similarity, spreading activation, and node strength
            relevance_score:
              (similarityMap.get(n.id) ?? 0) * 0.5 +
              n.activation * 0.35 +
              n.strength * 0.15,
            semantic_similarity: similarityMap.get(n.id) ?? 0,
            activation: n.activation,
            strength: n.strength,
          }))
          .sort((a, b) => b.relevance_score - a.relevance_score)
          .slice(0, top_k);

        // Include 1-hop neighbors for each result so small models can
        // read "name → Pablo" without needing a second tool call.
        const nodeMap = new Map(allNodes.map((n) => [n.id, n.label]));
        const allEdgesForRecall = getAllEdges();
        const adjacency = new Map<string, Array<{ label: string; weight: number }>>();
        for (const e of allEdgesForRecall) {
          if (!adjacency.has(e.from_id)) adjacency.set(e.from_id, []);
          if (!adjacency.has(e.to_id))   adjacency.set(e.to_id,   []);
          const fromLabel = nodeMap.get(e.to_id);
          const toLabel   = nodeMap.get(e.from_id);
          if (fromLabel && !isContextNode(fromLabel))
            adjacency.get(e.from_id)!.push({ label: fromLabel, weight: e.weight });
          if (toLabel && !isContextNode(toLabel))
            adjacency.get(e.to_id)!.push({ label: toLabel, weight: e.weight });
        }

        const rankedWithNeighbors = ranked.map((n) => ({
          ...n,
          connected_to: (adjacency.get(n.id) ?? [])
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 5)
            .map((nb) => nb.label),
        }));

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  query,
                  context: activeContext,
                  results: rankedWithNeighbors,
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "memory_consolidate": {
        const stats = consolidate();
        return {
          content: [{ type: "text", text: JSON.stringify({ consolidation: stats }, null, 2) }],
        };
      }

      case "memory_maintenance": {
        const force = (args as any)?.force === true;
        const report = runMaintenance(force);
        return {
          content: [{ type: "text", text: JSON.stringify(report, null, 2) }],
        };
      }

      case "memory_status": {
        const stats = getStats();
        const topNodes = getTopNodes(10);
        const activeContext = getActiveContext();

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  active_context: activeContext,
                  total_nodes: stats.nodeCount,
                  total_edges: stats.edgeCount,
                  last_consolidation: stats.lastConsolidation
                    ? new Date(stats.lastConsolidation).toISOString()
                    : null,
                  top_10_strongest: topNodes.map((n) => ({
                    label: n.label,
                    strength: n.strength,
                    access_count: n.access_count,
                    last_accessed: new Date(n.last_accessed_at).toISOString(),
                  })),
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "memory_set_context": {
        const contextArg = (args as any).context as string | undefined;
        const context = contextArg?.trim() ? contextArg.trim() : detectContext();

        setMeta("active_context", context);
        const contextNodeId = await ensureContextNode(context);

        // Activate context node to prime it
        await spreadActivation([{ id: contextNodeId, activation: 1.0 }], 2, 0.5, 0.1);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ active_context: context, context_node_id: contextNodeId }, null, 2),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (err: any) {
    return {
      content: [{ type: "text", text: JSON.stringify({ error: err.message }) }],
      isError: true,
    };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  process.stderr.write("memory-mcp v1.1 running\n");
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err.message}\n`);
  process.exit(1);
});
