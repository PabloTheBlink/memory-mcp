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
  updateNodeImportance,
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
      description: "Activate a concept and its context. Returns the activated subgraph. Use this when learning something new.",
      inputSchema: {
        type: "object",
        properties: {
          concept: { type: "string", description: "The concept to activate" },
          importance: { type: "number", description: "0-1 significance", default: 0.5 },
        },
        required: ["concept"],
      },
    },
    {
      name: "memory_associate",
      description: "Link two concepts. Types: causal, temporal, semantic, episodic.",
      inputSchema: {
        type: "object",
        properties: {
          concept_a: { type: "string" },
          concept_b: { type: "string" },
          type: {
            type: "string",
            enum: ["causal", "temporal", "semantic", "episodic", "abstraction"],
            default: "semantic",
          },
          strength: { type: "number", minimum: 0, maximum: 1 },
        },
        required: ["concept_a", "concept_b"],
      },
    },
    {
      name: "memory_recall",
      description: "Search memories by semantic similarity and spreading activation.",
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
      description: "Prune weak memories and apply Ebbinghaus decay.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "memory_status",
      description: "Show system stats and active context.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "memory_maintenance",
      description: "Run full maintenance (decay, link, merge, prune). Rate-limited to 1/hour.",
      inputSchema: {
        type: "object",
        properties: {
          force: { type: "boolean", default: false },
        },
      },
    },
    {
      name: "memory_set_context",
      description: "Change active context (e.g. 'user', 'project:name'). Bias recall toward this context.",
      inputSchema: {
        type: "object",
        properties: {
          context: {
            type: "string",
            description: "Context label. Leave empty to auto-detect.",
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
        const { concept, importance = 0.5 } = args as any;
        const activeContext = getActiveContext();

        let node = findOrCreateNode(concept, null, importance);
        if (!node.embedding) {
          const emb = await getEmbedding(concept);
          updateNodeEmbedding(node.id, emb);
          node = { ...node, embedding: emb };
        }
        
        // Update importance if it changed significantly
        if (importance !== 0.5 && Math.abs(node.importance - importance) > 0.1) {
          updateNodeImportance(node.id, (node.importance + importance) / 2);
        }

        // Find semantically similar nodes and wire them
        const allNodes = getAllNodes();
        const similar = findSimilar(node.embedding!, allNodes, 0.78, 20).filter((s) => s.id !== node.id);
        for (const s of similar.slice(0, 4)) {
          upsertEdge(node.id, s.id, "semantic", s.similarity * 0.12);
        }

        // Human-like: Temporal Co-occurrence (Episodic memory)
        // Link to recently activated nodes in this context
        const lastActivatedId = getMeta(`last_act_${activeContext}`);
        if (lastActivatedId && lastActivatedId !== node.id) {
          upsertEdge(node.id, lastActivatedId, "temporal", 0.25);
        }
        setMeta(`last_act_${activeContext}`, node.id);

        // Bind to active context
        const contextNodeId = await ensureContextNode(activeContext);
        bindToContext(node.id, contextNodeId);

        touchNode(node.id);

        // Spread activation (Higher depth for initialization)
        const result = await spreadActivation(
          [
            { id: node.id, activation: 1.0 },
            { id: contextNodeId, activation: 0.4 },
          ],
          3,
          0.5,
          0.06
        );

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                  concept,
                  context: activeContext,
                  subgraph: {
                    nodes: result.nodes
                      .filter((n) => !isContextNode(n.label))
                      .map((n) => ({ label: n.label, strength: Math.round(n.strength * 100) / 100 })),
                    edges: result.edges.slice(0, 10).map(e => {
                      const from = result.nodes.find(n => n.id === e.from_id)?.label;
                      const to = result.nodes.find(n => n.id === e.to_id)?.label;
                      return `${from} --(${e.type})--> ${to}`;
                    }),
                  },
                }),
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

        const boost = strength !== undefined ? strength * 0.25 : 0.15;
        upsertEdge(nodeA.id, nodeB.id, type, boost);

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
              text: JSON.stringify({
                  associated: `${concept_a} <-> ${concept_b}`,
                  type,
                  context: activeContext,
                }),
            },
          ],
        };
      }

      case "memory_recall": {
        const { query, top_k = 10 } = args as any;
        const activeContext = getActiveContext();

        const queryEmbedding = await getEmbedding(query);
        const allNodes = getAllNodes();
        
        // 0.32 threshold for cross-language recall
        const similar = findSimilar(queryEmbedding, allNodes, 0.30, top_k * 4);

        // Human-like refinement: Identify "hidden" associations (Language Agnostic)
        const STOP_WORDS = new Set(["the", "and", "for", "with", "that", "this", "los", "las", "con", "para", "que", "una", "uno", "del", "por", "des", "les", "une", "est"]);
        const queryLower = query.toLowerCase();
        const queryTokens = queryLower.split(/[\s,.;:!?]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));

        const refinedSimilar = similar.map(s => {
          const node = allNodes.find(n => n.id === s.id);
          if (!node) return s;
          const labelLower = node.label.toLowerCase();
          const labelTokens = labelLower.split(/[\s,.;:!?]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));
          
          const hasLexicalOverlap = labelTokens.some(t => queryTokens.includes(t)) || 
                                   (labelLower.length > 3 && queryLower.includes(labelLower));
          
          // Boost if semantically strong but lexically different (likely translation/deep concept)
          const insightBoost = (!hasLexicalOverlap && s.similarity > 0.65) ? 0.20 : 0;
          // Importance also helps recall
          const importanceBoost = (node.importance || 0.5) * 0.1;

          return { ...s, similarity: Math.min(1.0, s.similarity + insightBoost + importanceBoost) };
        }).sort((a, b) => b.similarity - a.similarity);

        const contextNodeId = await ensureContextNode(activeContext);
        const seeds: Array<{ id: string; activation: number }> = [
          { id: contextNodeId, activation: 0.5 }, // Strong context priming
          ...refinedSimilar.slice(0, 6).map((s) => ({ id: s.id, activation: 1.0 })),
        ];

        if (seeds.length === 1) {
          return {
            content: [{ type: "text", text: JSON.stringify({ query, context: activeContext, results: [] }) }],
          };
        }

        for (const { id } of seeds) touchNode(id);
        const result = await spreadActivation(seeds, 3, 0.48, 0.05);

        const similarityMap = new Map(refinedSimilar.map((s) => [s.id, s.similarity]));

        const ranked = result.nodes
          .filter((n) => !isContextNode(n.label))
          .map((n) => ({
            id: n.id,
            label: n.label,
            relevance_score:
              (similarityMap.get(n.id) ?? 0) * 0.45 + 
              n.activation * 0.35 +                 
              n.strength * 0.12 +                   
              n.importance * 0.08,                  
            connected_to: [] as string[]
          }))
          .sort((a, b) => b.relevance_score - a.relevance_score)
          .slice(0, top_k);

        // Build adjacency
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

        // Reinforce associations (Long Term Potentiation)
        const contextNodeIdForReinforcement = await ensureContextNode(activeContext);
        for (const n of ranked.slice(0, 3)) {
          upsertEdge(n.id, contextNodeIdForReinforcement, "episodic", 0.1);
          n.connected_to = (adjacency.get(n.id) ?? [])
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 5)
            .map((nb) => nb.label);
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                  query,
                  context: activeContext,
                  results: ranked.map(r => ({
                    label: r.label,
                    relevance: Math.round(r.relevance_score * 100) / 100,
                    connected: r.connected_to
                  })),
                }),
            },
          ],
        };
      }

      case "memory_consolidate": {
        const stats = consolidate();
        return {
          content: [{ type: "text", text: JSON.stringify({ consolidation: stats }) }],
        };
      }

      case "memory_maintenance": {
        const force = (args as any)?.force === true;
        const report = runMaintenance(force);
        return {
          content: [{ type: "text", text: JSON.stringify(report) }],
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
              text: JSON.stringify({
                  context: activeContext,
                  stats: { nodes: stats.nodeCount, edges: stats.edgeCount },
                  top_memories: topNodes.map((n) => n.label),
                }),
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
              text: JSON.stringify({ context }),
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
