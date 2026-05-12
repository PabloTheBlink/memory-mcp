import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

// Intercept stdout to prevent non-JSON messages (from wrappers like tsx) from breaking the MCP protocol
const originalStdoutWrite = process.stdout.write.bind(process.stdout);
(process.stdout.write as any) = (chunk: any, encoding: any, callback: any) => {
  const str = typeof chunk === 'string' ? chunk : chunk.toString();
  if (str.trim().startsWith('{') || str.trim().startsWith('[')) {
    return originalStdoutWrite(chunk, encoding, callback);
  }
  return process.stderr.write(chunk, encoding, callback);
};

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
  logEvent,
  updateNodeLabel,
  searchNodesByLabel,
  deleteNode,
} from "./graph";
import { getEmbedding, getEmbeddings, findSimilar, cosineSimilarity } from "./embeddings";
import { spreadActivation } from "./activation";
import { runMaintenance } from "./maintenance";
import {
  getActiveContext,
  ensureContextNode,
  bindToContext,
  contextLabel,
  isContextNode,
  detectContext,
  getDeviceId,
  setActiveContext,
} from "./context";

const EMBEDDING_MODEL_ID = "Xenova/all-MiniLM-L6-v2";

async function runReindexingIfNeeded() {
  const storedModel = await getMeta("embedding_model");
  const allNodes = await getAllNodes();
  const nodesToFix = allNodes.filter(n => n.embedding === null);

  if (storedModel === EMBEDDING_MODEL_ID && nodesToFix.length === 0) return;

  if (storedModel !== EMBEDDING_MODEL_ID) {
    process.stderr.write(`Re-indexing memory: switching from ${storedModel || "unknown"} to ${EMBEDDING_MODEL_ID}...\n`);
  } else if (nodesToFix.length > 0) {
    process.stderr.write(`Fixing ${nodesToFix.length} nodes with missing embeddings...\n`);
  }

  const nodes = (storedModel !== EMBEDDING_MODEL_ID) ? allNodes : nodesToFix;
  process.stderr.write(`Processing ${nodes.length} nodes...\n`);
  
  const CHUNK_SIZE = 20;
  for (let i = 0; i < nodes.length; i += CHUNK_SIZE) {
    const chunk = nodes.slice(i, i + CHUNK_SIZE);
    try {
      const embeddings = await getEmbeddings(chunk.map(n => n.label));
      for (let j = 0; j < chunk.length; j++) {
        await updateNodeEmbedding(chunk[j].id, embeddings[j]);
      }
      if (i % 40 === 0) process.stderr.write(`Progress: ${i}/${nodes.length}\n`);
    } catch (e) {
      process.stderr.write(`Failed to index batch at ${i}: ${e}\n`);
    }
  }

  await setMeta("embedding_model", EMBEDDING_MODEL_ID);
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
    {
      "name": "memory_suggest",
      "description": "Proactively suggests contextually relevant memories based on current activity. Use this when you need inspiration or to discover hidden associations.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "limit": { "type": "number", "default": 5 }
        }
      },
    },
    {
      "name": "memory_get_context_summary",
      "description": "Returns a high-level summary of the current active context, including main conceptual hubs and recent activity.",
      "inputSchema": {
        "type": "object",
        "properties": {}
      },
    },
    {
      "name": "memory_update_node",
      "description": "Refine, rename or forget a concept. Use this to maintain memory accuracy and remove errors.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "label": { "type": "string", "description": "The current label of the node" },
          "new_label": { "type": "string", "description": "Optional new label to rename the node" },
          "importance": { "type": "number", "description": "Update importance (0-1)" },
          "forget": { "type": "boolean", "description": "If true, permanently deletes the node and its connections" }
        },
        "required": ["label"]
      }
    },
    {
      "name": "memory_list_hubs",
      "description": "List existing projects, contexts and high-level conceptual hubs.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "type": { "type": "string", "enum": ["projects", "contexts", "all"], "default": "all" }
        }
      }
    },
    {
      "name": "memory_activate_batch",
      "description": "Efficiently activate multiple concepts at once. Perfect for atomizing documents or project setups.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "concepts": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "concept": { "type": "string" },
                "importance": { "type": "number", "default": 0.5 }
              },
              "required": ["concept"]
            }
          }
        },
        "required": ["concepts"]
      }
    },
  ],
})),

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "memory_learn_rule": {
        const { rule, context: ruleCtx } = args as any;
        const deviceId = getDeviceId();
        const activeContext = ruleCtx || await getActiveContext();
        const label = `rule:${rule}`;

        let node = await findOrCreateNode(label, null, 1.0, null, deviceId, "private"); // Rules are usually private
        if (!node.embedding) {
          const emb = await getEmbedding(label);
          await updateNodeEmbedding(node.id, emb);
        }

        const contextNodeId = await ensureContextNode(activeContext);
        await bindToContext(node.id, contextNodeId);

        // Bind to device context
        const deviceNodeId = await ensureContextNode(`device:${deviceId}`);
        await bindToContext(node.id, deviceNodeId);

        await touchNode(node.id);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ 
                learned: rule, 
                context: activeContext,
                user_id: deviceId,
                message: "Rule stored. I will recall this whenever relevant tasks arise to save you from repeating it." 
              }),
            },
          ],
        };
      }

      case "memory_activate": {
        const { concept, importance = 0.5 } = args as any;
        const deviceId = getDeviceId();
        await logEvent(`Activating concept: ${concept}`, deviceId);
        const activeContext = await getActiveContext();
        
        // Projects and conceptual hubs are shared by default
        const visibility = (concept.startsWith("project:") || concept.startsWith("concept:")) ? "shared" : "private";

        let node = await findOrCreateNode(concept, null, importance, null, deviceId, visibility);
        if (!node.embedding) {
          const emb = await getEmbedding(concept);
          await updateNodeEmbedding(node.id, emb);
          node = { ...node, embedding: emb };
        }
        
        // Update importance if it changed significantly
        if (importance !== 0.5 && Math.abs(node.importance - importance) > 0.1) {
          await updateNodeImportance(node.id, (node.importance + importance) / 2);
        }

        // Find semantically similar nodes and wire them
        const allNodes = await getAllNodes(deviceId);
        const similar = findSimilar(node.embedding!, allNodes, 0.78, 20).filter((s) => s.id !== node.id);
        for (const s of similar.slice(0, 4)) {
          await upsertEdge(node.id, s.id, "semantic", s.similarity * 0.12, deviceId);
        }

        // Human-like: Temporal Co-occurrence (Episodic memory)
        // Link to recently activated nodes in this context
        const lastActivatedId = await getMeta(`last_act_${activeContext}_${deviceId}`);
        if (lastActivatedId && lastActivatedId !== node.id) {
          await upsertEdge(node.id, lastActivatedId, "temporal", 0.25, deviceId);
        }
        
        // NEW: Proactive Associative Linking
        // Link to other currently "hot" nodes in this context to build a dense local web
        const now = Date.now();
        const topFired = allNodes
          .filter(n => n.id !== node.id && (now - n.last_fired_at) < 10 * 60 * 1000)
          .sort((a, b) => b.last_fired_at - a.last_fired_at)
          .slice(0, 2);
        for (const hot of topFired) {
          await upsertEdge(node.id, hot.id, "episodic", 0.15, deviceId);
        }

        await setMeta(`last_act_${activeContext}_${deviceId}`, node.id);

        // Bind to active context
        const contextNodeId = await ensureContextNode(activeContext);
        await bindToContext(node.id, contextNodeId);

        // Bind to device context for multi-user identification
        const deviceNodeId = await ensureContextNode(`device:${deviceId}`);
        await bindToContext(node.id, deviceNodeId);

        await touchNode(node.id);
        await fireNode(node.id);

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
        const deviceId = getDeviceId();
        const activeContext = await getActiveContext();

        // Determine visibility based on content
        const visibility = (concept_a.startsWith("project:") || concept_b.startsWith("project:")) ? "shared" : "private";

        let nodeA = await findOrCreateNode(concept_a, null, 0.5, null, deviceId, visibility);
        let nodeB = await findOrCreateNode(concept_b, null, 0.5, null, deviceId, visibility);

        if (!nodeA.embedding) {
          const emb = await getEmbedding(concept_a);
          await updateNodeEmbedding(nodeA.id, emb);
        }
        if (!nodeB.embedding) {
          const emb = await getEmbedding(concept_b);
          await updateNodeEmbedding(nodeB.id, emb);
        }

        const boost = strength !== undefined ? strength * 0.25 : 0.15;
        await upsertEdge(nodeA.id, nodeB.id, type, boost, deviceId);
        await logEvent(`Associated ${concept_a} with ${concept_b} (${type})`, deviceId);

        // Bind both to context
        const contextNodeId = await ensureContextNode(activeContext);
        bindToContext(nodeA.id, contextNodeId);
        bindToContext(nodeB.id, contextNodeId);

        // Bind to device context
        const deviceNodeId = await ensureContextNode(`device:${deviceId}`);
        bindToContext(nodeA.id, deviceNodeId);
        bindToContext(nodeB.id, deviceNodeId);

        await touchNode(nodeA.id);
        await touchNode(nodeB.id);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                  associated: `${concept_a} <-> ${concept_b}`,
                  type,
                  visibility,
                  context: activeContext,
                }),
            },
          ],
        };
      }

      case "memory_recall": {
        const { query, top_k = 10 } = args as any;
        const deviceId = getDeviceId();
        const activeContext = await getActiveContext();
        await logEvent(`Recalling memory for query: ${query}`, deviceId);

        const queryEmbedding = await getEmbedding(query);
        const allNodes = await getAllNodes(deviceId);
        
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
        const deviceNodeId = await ensureContextNode(`device:${deviceId}`);
        const seeds: Array<{ id: string; activation: number }> = [
          { id: contextNodeId, activation: 0.4 }, // Strong context priming
          { id: deviceNodeId, activation: 0.3 },  // Device priming for identity/preferences
          ...refinedSimilar.slice(0, 6).map((s) => ({ id: s.id, activation: 1.0 })),
        ];

        if (seeds.length === 1 && rules.length === 0) {
          return {
            content: [{ type: "text", text: JSON.stringify({ query, context: activeContext, results: [] }) }],
          };
        }

        for (const { id } of seeds) {
          await touchNode(id);
          await fireNode(id);
        }
        // Also fire the top results to make them flash
        for (const n of refinedSimilar.slice(0, 10)) {
          await fireNode(n.id);
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
        const allEdgesForRecall = await getAllEdges(deviceId);
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
          await upsertEdge(n.id, contextNodeIdForReinforcement, "episodic", 0.1, deviceId);
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


      case "memory_maintenance": {
        const force = (args as any)?.force === true;
        const report = await runMaintenance(force);
        return {
          content: [{ type: "text", text: JSON.stringify(report) }],
        };
      }

      case "memory_status": {
        const deviceId = getDeviceId();
        const stats = await getStats(deviceId);
        const topNodes = await getTopNodes(10, deviceId);
        const activeContext = await getActiveContext();

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                  context: activeContext,
                  device_id: deviceId,
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

        await setActiveContext(context);
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
        const startNode = await findOrCreateNode(start_concept);
        
        const path: string[] = [startNode.label];
        const visited = new Set<string>([startNode.id]);
        let currentId = startNode.id;

        for (let i = 0; i < depth; i++) {
          const neighbors = await getNeighbors(currentId);
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

      case "memory_suggest": {
        const { limit = 5 } = args as any;
        const deviceId = getDeviceId();
        const activeContext = await getActiveContext();
        const contextNodeId = await ensureContextNode(activeContext);
        const deviceNodeId = await ensureContextNode(`device:${deviceId}`);

        // Seed with currently fired nodes or active context
        const allNodes = await getAllNodes(deviceId);
        const fired = allNodes
          .filter(n => (Date.now() - n.last_fired_at) < 15 * 60 * 1000)
          .map(n => ({ id: n.id, activation: 1.0 }));
        
        const seeds = fired.length > 0 ? fired : [{ id: contextNodeId, activation: 1.0 }];
        
        const result = await spreadActivation(seeds, 3, 0.45, 0.08, contextNodeId, true);
        const suggestions = result.nodes
          .filter(n => !isContextNode(n.label) && !fired.some(f => f.id === n.id))
          .sort((a, b) => b.activation - a.activation)
          .slice(0, limit)
          .map(n => ({
             label: n.label,
             reason: "Associative resonance with current activity"
          }));

        return {
          content: [{ type: "text", text: JSON.stringify({ suggestions, context: activeContext }) }],
        };
      }

      case "memory_get_context_summary": {
        const deviceId = getDeviceId();
        const activeContext = await getActiveContext();
        const contextNodeId = await ensureContextNode(activeContext);
        
        const neighbors = await getNeighbors(contextNodeId, deviceId);
        const hubs = neighbors
          .filter(n => n.node.label.startsWith("concept:") || n.node.importance > 0.8)
          .map(n => n.node.label);
        
        const recent = neighbors
          .sort((a, b) => b.node.last_accessed_at - a.node.last_accessed_at)
          .slice(0, 10)
          .map(n => n.node.label);

        return {
          content: [{ 
            type: "text", 
            text: JSON.stringify({ 
              context: activeContext,
              conceptual_hubs: hubs,
              recently_accessed: recent,
              total_context_nodes: neighbors.length
            }) 
          }],
        };
      }

      case "memory_update_node": {
        const { label, new_label, importance, forget } = args as any;
        const deviceId = getDeviceId();
        const node = (await getAllNodes(deviceId)).find(n => n.label === label);

        if (!node) {
          throw new Error(`Node not found: ${label}`);
        }

        if (forget) {
          await deleteNode(node.id);
          await logEvent(`Forgot node: ${label}`, deviceId);
          return { content: [{ type: "text", text: JSON.stringify({ status: "forgotten", label }) }] };
        }

        if (new_label) {
          await updateNodeLabel(node.id, new_label);
          // Re-embed if label changed
          const emb = await getEmbedding(new_label);
          await updateNodeEmbedding(node.id, emb);
          await logEvent(`Renamed node: ${label} -> ${new_label}`, deviceId);
        }

        if (importance !== undefined) {
          await updateNodeImportance(node.id, importance);
        }

        return {
          content: [{ type: "text", text: JSON.stringify({ status: "updated", label: new_label || label }) }],
        };
      }

      case "memory_list_hubs": {
        const { type = "all" } = args as any;
        const deviceId = getDeviceId();
        let hubs: string[] = [];

        if (type === "projects" || type === "all") {
          const nodes = await searchNodesByLabel("project:%", deviceId);
          hubs.push(...nodes.map(n => n.label));
        }
        if (type === "contexts" || type === "all") {
          const nodes = await searchNodesByLabel("[ctx:%", deviceId);
          hubs.push(...nodes.map(n => n.label));
        }
        
        // Add high-importance concepts
        const topNodes = await getTopNodes(20, deviceId);
        hubs.push(...topNodes.filter(n => n.importance > 0.8 && !hubs.includes(n.label)).map(n => n.label));

        return {
          content: [{ type: "text", text: JSON.stringify({ hubs: Array.from(new Set(hubs)) }) }],
        };
      }

      case "memory_activate_batch": {
        const { concepts } = args as any;
        const deviceId = getDeviceId();
        const activeContext = await getActiveContext();
        const contextNodeId = await ensureContextNode(activeContext);
        
        const results = [];
        for (const item of concepts) {
          const { concept, importance = 0.5 } = item;
          const node = await findOrCreateNode(concept, null, importance, null, deviceId, "private");
          if (!node.embedding) {
             const emb = await getEmbedding(concept);
             await updateNodeEmbedding(node.id, emb);
          }
          await bindToContext(node.id, contextNodeId);
          await touchNode(node.id);
          results.push(concept);
        }

        await logEvent(`Batch activation of ${concepts.length} concepts`, deviceId);

        return {
          content: [{ type: "text", text: JSON.stringify({ activated: results, context: activeContext }) }],
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
  // process.stderr.write("memory-mcp v1.1 running (local embeddings)\n");
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err.message}\n`);
  process.exit(1);
});
