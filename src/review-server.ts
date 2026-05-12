import http from "http";
import fs from "fs";
import path from "path";
import {
  findOrCreateNode,
  getAllNodes,
  getAllEdges,
  getNodeById,
  deleteNode,
  deleteEdge,
  MemoryNode,
  MemoryEdge,
  rewireEdges,
} from "./graph";
import { getDb } from "./db";
import { getEmbedding, findSimilar, cosineSimilarity } from "./embeddings";
import { spreadActivation } from "./activation";
import { getActiveContext, ensureContextNode, getDeviceId } from "./context";

const PORT = 3131;

// --- DB helpers not in graph.ts ---

async function renameNode(id: string, newLabel: string): Promise<void> {
  const db = await getDb();
  await db.run("UPDATE nodes SET label = ? WHERE id = ?", [newLabel, id]);
}


// --- Context extraction from episodic edges ---

async function nodeContexts(): Promise<Map<string, string[]>> {
  const edges = await getAllEdges();
  const nodes = await getAllNodes();
  const ctxLabels = new Map(
    nodes.filter((n) => n.label.startsWith("[ctx:")).map((n) => [n.id, n.label.slice(5, -1)])
  );
  const map = new Map<string, string[]>();
  for (const e of edges) {
    if (e.type === "episodic") {
      const ctxId = ctxLabels.has(e.to_id) ? e.to_id : ctxLabels.has(e.from_id) ? e.from_id : null;
      const memId = ctxLabels.has(e.to_id) ? e.from_id : ctxLabels.has(e.from_id) ? e.to_id : null;
      if (ctxId && memId) {
        const ctx = ctxLabels.get(ctxId)!;
        const existing = map.get(memId) ?? [];
        if (!existing.includes(ctx)) map.set(memId, [...existing, ctx]);
      }
    }
  }
  return map;
}

// --- Duplicate detection ---

interface DuplicateCandidate {
  a: MemoryNode;
  b: MemoryNode;
  similarity: number;
  likelyFalsePositive: boolean;
  reason: string;
}

function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
  return dp[m][n];
}

async function findDuplicates(threshold = 0.92): Promise<DuplicateCandidate[]> {
  const nodes = (await getAllNodes()).filter((n) => n.embedding && !n.label.startsWith("[ctx:"));
  const results: DuplicateCandidate[] = [];

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const sim = cosineSimilarity(nodes[i].embedding!, nodes[j].embedding!);
      if (sim < threshold) continue;

      const la = nodes[i].label.toLowerCase();
      const lb = nodes[j].label.toLowerCase();
      const lev = levenshtein(la, lb);
      const maxLen = Math.max(la.length, lb.length);
      const textSimilarity = 1 - lev / maxLen;

      const likelyFalsePositive = textSimilarity < 0.35;

      let reason = "";
      if (likelyFalsePositive) {
        reason = `Embedding artifact — labels share no text similarity (${Math.round(textSimilarity * 100)}% text match)`;
      } else if (textSimilarity > 0.8) {
        reason = `Very similar text (${Math.round(textSimilarity * 100)}% text match)`;
      } else {
        reason = `Same semantic meaning, different wording`;
      }

      results.push({ a: nodes[i], b: nodes[j], similarity: sim, likelyFalsePositive, reason });
    }
  }

  return results.sort((a, b) => {
    if (a.likelyFalsePositive !== b.likelyFalsePositive)
      return a.likelyFalsePositive ? 1 : -1;
    return b.similarity - a.similarity;
  });
}

// --- API Helpers ---

function json(res: http.ServerResponse, data: object, status = 200) {
  res.writeHead(status, { 
    "Content-Type": "application/json",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0"
  });
  res.end(JSON.stringify(data));
}

async function readBody(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => { try { resolve(JSON.parse(data)); } catch { resolve({}); } });
  });
}

async function buildGraphData() {
  const allNodes = await getAllNodes();
  const allEdges = await getAllEdges();
  const ctxMap = await nodeContexts();

  const ctxNames = Array.from(new Set(
    allNodes.filter(n => n.label.startsWith("[ctx:")).map(n => n.label.slice(5, -1))
  ));

  const nodes = allNodes.map(n => ({
    id: n.id,
    label: n.label,
    isContext: n.label.startsWith("[ctx:"),
    isHub: n.label.startsWith("concept:"),
    contextName: n.label.startsWith("[ctx:") ? n.label.slice(5, -1) : null,
    contexts: ctxMap.get(n.id) ?? [],
    strength: n.strength,
    importance: n.importance,
    access_count: n.access_count,
    last_accessed_at: n.last_accessed_at,
    last_fired_at: n.last_fired_at,
    last_suggested_at: n.last_suggested_at || 0,
    metadata: n.metadata,
    user_id: n.user_id,
    visibility: n.visibility,
  }));

  const nodeIds = new Set(nodes.map(n => n.id));
  const links = allEdges
    .filter(e => nodeIds.has(e.from_id) && nodeIds.has(e.to_id))
    .map(e => ({
      source: e.from_id,
      target: e.to_id,
      weight: e.weight,
      type: e.type,
      co_occurrences: e.co_occurrences,
      user_id: e.user_id,
    }));

  return { nodes, links, ctxNames };
}

const server = http.createServer(async (req, res) => {
  const url = req.url ?? "/";

  if (req.method === "GET" && url === "/") {
    try {
      const htmlPath = path.join(__dirname, "review-dashboard.html");
      const html = fs.readFileSync(htmlPath, "utf8");
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(html);
    } catch (e) {
      res.writeHead(500);
      res.end("Error loading dashboard. Ensure src/review-dashboard.html exists.");
    }
    return;
  }

  if (req.method === "GET" && url === "/api/graph") {
    json(res, await buildGraphData());
    return;
  }

  if (req.method === "GET" && url === "/api/duplicates") {
    json(res, await findDuplicates());
    return;
  }

  if (req.method === "GET" && url === "/api/logs") {
    const db = await getDb();
    const logs = await db.queryAll("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 50");
    json(res, logs);
    return;
  }

  if (req.method === "GET" && url === "/api/stats") {
    const db = await getDb();
    const nodes = await db.queryGet("SELECT COUNT(*) as c FROM nodes");
    const edges = await db.queryGet("SELECT COUNT(*) as c FROM edges");
    json(res, { nodes: nodes.c, edges: edges.c });
    return;
  }

  if (req.method === "GET" && url === "/api/suggest") {
    const deviceId = getDeviceId();
    const activeContext = await getActiveContext();
    const contextNodeId = await ensureContextNode(activeContext);
    
    const allNodes = await getAllNodes(deviceId);
    const fired = allNodes
      .filter(n => (Date.now() - n.last_fired_at) < 15 * 60 * 1000)
      .map(n => ({ id: n.id, activation: 1.0 }));
    
    const seeds = fired.length > 0 ? fired : [{ id: contextNodeId, activation: 1.0 }];
    const result = await spreadActivation(seeds, 5, 0.45, 0.01, contextNodeId, true);
    
    const suggestions = result.nodes
      .filter(n => !n.label.startsWith("[ctx:") && !fired.some(f => f.id === n.id))
      .sort((a, b) => b.activation - a.activation)
      .slice(0, 5)
      .map(n => ({ label: n.label, activation: n.activation }));
    
    json(res, { suggestions, context: activeContext });
    return;
  }

  if (req.method === "GET" && url === "/api/context-summary") {
    const deviceId = getDeviceId();
    const activeContext = await getActiveContext();
    const contextNodeId = await ensureContextNode(activeContext);
    
    const nodes = await getAllNodes(deviceId);
    const contextNode = nodes.find(n => n.id === contextNodeId);
    
    const edges = await getAllEdges(deviceId);
    const neighbors = edges
      .filter(e => e.from_id === contextNodeId || e.to_id === contextNodeId)
      .map(e => {
        const otherId = e.from_id === contextNodeId ? e.to_id : e.from_id;
        const otherNode = nodes.find(n => n.id === otherId);
        return { label: otherNode?.label, weight: e.weight };
      })
      .filter(n => n.label);

    const hubs = neighbors
      .filter(n => n.label!.startsWith("concept:") || n.weight > 0.8)
      .map(n => n.label);
    
    json(res, { 
      context: activeContext,
      conceptual_hubs: hubs,
      total_context_nodes: neighbors.length
    });
    return;
  }

  if (req.method === "POST") {
    const body = await readBody(req);

    if (url === "/delete-node") {
      try { await deleteNode(body.id); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/delete-edge") {
      try { await deleteEdge(body.from_id, body.to_id); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/rename") {
      try { await renameNode(body.id, body.label); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/merge") {
      try {
        await rewireEdges(body.delete_id, body.keep_id);
        await deleteNode(body.delete_id);
        json(res, { ok: true });
      } catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }
  }

  res.writeHead(404);
  res.end();
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Memory Review → http://localhost:${PORT}`);
});
