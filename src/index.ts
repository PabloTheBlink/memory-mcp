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
  fireNode,
  setMeta,
  getMeta,
  getNeighbors,
} from "./graph";
import { getEmbedding, getEmbeddings, findSimilar, cosineSimilarity } from "./embeddings";
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

const EMBEDDING_MODEL_ID = "Xenova/all-MiniLM-L6-v2";

async function runReindexingIfNeeded() {
  const storedModel = getMeta("embedding_model");
  if (storedModel === EMBEDDING_MODEL_ID) return;

  process.stderr.write(`Re-indexing memory: switching from ${storedModel || "unknown"} to ${EMBEDDING_MODEL_ID}...\n`);

  const nodes = getAllNodes();
  process.stderr.write(`Re-indexing ${nodes.length} nodes...\n`);
  
  const CHUNK_SIZE = 20;
  for (let i = 0; i < nodes.length; i += CHUNK_SIZE) {
    const chunk = nodes.slice(i, i + CHUNK_SIZE);
    try {
      const embeddings = await getEmbeddings(chunk.map(n => n.label));
      for (let j = 0; j < chunk.length; j++) {
        updateNodeEmbedding(chunk[j].id, embeddings[j]);
      }
      if (i % 40 === 0) process.stderr.write(`Progress: ${i}/${nodes.length}\n`);
    } catch (e) {
      process.stderr.write(`Failed to index batch at ${i}: ${e}\n`);
    }
  }

  setMeta("embedding_model", EMBEDDING_MODEL_ID);
  process.stderr.write(`Re-indexing complete. Nodes updated.\n`);
}

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
      description: "MANDATORIO: Ejecutar al inicio de cada sesión con una consulta sobre 'preferencias del usuario' y 'reglas de estilo' para recuperar el contexto cognitivo. Busca memorias por similitud semántica y propagación de activación.",
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
      description: "MANDATORIO: Ejecutar SIEMPRE al inicio de una nueva sesión para identificar el contexto activo, estadísticas del sistema y nodos principales de memoria.",
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
    {
      name: "memory_learn_rule",
      description: "Store a permanent rule or preference (e.g. 'Always use pnpm', 'Prefer compact code'). This ensures future agent actions align with your style without repeating instructions.",
      inputSchema: {
        type: "object",
        properties: {
          rule: { type: "string", description: "The rule or preference to learn" },
          context: { type: "string", description: "Optional context for this rule" },
        },
        required: ["rule"],
      },
    },
    {
      "name": "memory_replay",
      "description": "Follow temporal and causal links to reconstruct a narrative or sequence of events starting from a concept.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "start_concept": { "type": "string" },
          "depth": { "type": "number", "default": 5 }
        },
        "required": ["start_concept"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "memory_learn_rule": {
        const { rule, context: ruleCtx } = args as any;
        const activeContext = ruleCtx || getActiveContext();
        const label = `rule:${rule}`;

        let node = findOrCreateNode(label, null, 1.0); // Rules are always important
        if (!node.embedding) {
          const emb = await getEmbedding(label);
          updateNodeEmbedding(node.id, emb);
        }

        const contextNodeId = await ensureContextNode(activeContext);
        bindToContext(node.id, contextNodeId);
        touchNode(node.id);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ 
                learned: rule, 
                context: activeContext,
                message: "Rule stored. I will recall this whenever relevant tasks arise to save you from repeating it." 
              }),
            },
          ],
        };
      }

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
        fireNode(node.id);

        // Spread activation (Higher depth for initialization)
        const result = await spreadActivation(
          [
            { id: node.id, activation: 1.0 },
            { id: contextNodeId, activation: 0.4 },
          ],
          3,
          0.5,
          0.06,
          contextNodeId
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
        const queryTokens = queryLower.split(/[\s,.;:!?]+/).filter((t: string) => t.length > 2 && !STOP_WORDS.has(t));

        const refinedSimilar = similar.map(s => {
          const node = allNodes.find(n => n.id === s.id);
          if (!node) return s;
          const labelLower = node.label.toLowerCase();
          const labelTokens = labelLower.split(/[\s,.;:!?]+/).filter((t: string) => t.length > 2 && !STOP_WORDS.has(t));
          
          const hasLexicalOverlap = labelTokens.some(t => queryTokens.includes(t)) || 
                                   (labelLower.length > 3 && queryLower.includes(labelLower));
          
          // Boost if semantically strong but lexically different (likely translation/deep concept)
          const insightBoost = (!hasLexicalOverlap && s.similarity > 0.65) ? 0.20 : 0;
          // Importance also helps recall
          const importanceBoost = (node.importance || 0.5) * 0.1;

          return { ...s, similarity: Math.min(1.0, s.similarity + insightBoost + importanceBoost) };
        }).sort((a, b) => b.similarity - a.similarity);

        // Learning: specifically look for rules and preferences
        const rules = allNodes
          .filter(n => n.label.startsWith("rule:") || n.label.startsWith("preference:"))
          .map(n => {
             const sim = n.embedding ? cosineSimilarity(queryEmbedding, n.embedding) : 0;
             return { label: n.label, sim };
          })
          .filter(r => r.sim > 0.35)
          .sort((a, b) => b.sim - a.sim)
          .slice(0, 5)
          .map(r => r.label);

        const contextNodeId = await ensureContextNode(activeContext);
        const seeds: Array<{ id: string; activation: number }> = [
          { id: contextNodeId, activation: 0.5 }, // Strong context priming
          ...refinedSimilar.slice(0, 6).map((s) => ({ id: s.id, activation: 1.0 })),
        ];

        if (seeds.length === 1 && rules.length === 0) {
          return {
            content: [{ type: "text", text: JSON.stringify({ query, context: activeContext, results: [] }) }],
          };
        }

        for (const { id } of seeds) {
          touchNode(id);
          fireNode(id);
        }
        // Also fire the top results to make them flash
        for (const n of refinedSimilar.slice(0, 10)) {
          fireNode(n.id);
        }
        const result = await spreadActivation(seeds, 3, 0.48, 0.05, contextNodeId);

        const similarityMap = new Map(refinedSimilar.map((s) => [s.id, s.similarity]));

        const ranked = result.nodes
          .filter((n) => !isContextNode(n.label) && !n.label.startsWith("rule:") && !n.label.startsWith("preference:"))
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
                  active_rules: rules,
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
        await spreadActivation([{ id: contextNodeId, activation: 1.0 }], 2, 0.5, 0.1, contextNodeId);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ context }),
            },
          ],
        };
      }

      case "memory_replay": {
        const { start_concept, depth = 5 } = args as any;
        const startNode = findOrCreateNode(start_concept);
        
        const path: string[] = [startNode.label];
        const visited = new Set<string>([startNode.id]);
        let currentId = startNode.id;

        for (let i = 0; i < depth; i++) {
          const neighbors = getNeighbors(currentId);
          // Prefer temporal or causal links for narrative flow
          const next = neighbors
            .filter((n: any) => !visited.has(n.node.id) && !isContextNode(n.node.label))
            .sort((a: any, b: any) => {
              const typeScore = (t: string) => t === "temporal" ? 3 : t === "causal" ? 2 : 1;
              return (typeScore(b.edge.type) * b.edge.weight) - (typeScore(a.edge.type) * a.edge.weight);
            })[0];

          if (!next) break;
          path.push(`${next.edge.type === "temporal" ? "then" : "leads to"} -> ${next.node.label}`);
          visited.add(next.node.id);
          currentId = next.node.id;
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ 
                start: start_concept,
                narrative: path 
              }),
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
  // Run migration if needed BEFORE starting the server
  await runReindexingIfNeeded();

  const transport = new StdioServerTransport();
  await server.connect(transport);
  process.stderr.write("memory-mcp v1.1 running (local embeddings)\n");
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err.message}\n`);
  process.exit(1);
});
