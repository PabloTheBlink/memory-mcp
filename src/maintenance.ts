/**
 * Standalone maintenance script. Can be run directly or triggered via hook.
 * Does not start the MCP server.
 *
 * Steps:
 *   1. Ebbinghaus decay on all nodes/edges
 *   2. Semantic linking — find node pairs with high embedding similarity but no edge
 *   3. Auto-merge — near-identical nodes (very high sim + text overlap)
 *   4. Orphan pruning — isolated nodes with low strength not seen recently
 */

import { getAllNodes, getAllEdges, getDb, upsertEdge, deleteNode, setMeta, getMeta } from "./graph";
import { cosineSimilarity } from "./embeddings";
import { consolidate } from "./decay";

const SEMANTIC_LINK_THRESHOLD  = 0.78;  // create edge if no edge exists
const SYNONYM_LINK_THRESHOLD   = 0.95;  // very strong link for synonyms/translations
const AUTO_MERGE_EMB_THRESHOLD = 0.98;  // merge if embedding sim this high...
const AUTO_MERGE_TEXT_THRESHOLD = 0.70; // ...AND text sim this high
const ORPHAN_MIN_AGE_DAYS       = 3;    // don't prune nodes touched recently
const ORPHAN_MAX_STRENGTH       = 0.15; // only prune if strength this low
const MIN_INTERVAL_HOURS        = 1;    // skip if run less than this ago

function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i-1] === b[j-1]
        ? dp[i-1][j-1]
        : 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
  return dp[m][n];
}

function textSimilarity(a: string, b: string): number {
  const la = a.toLowerCase(), lb = b.toLowerCase();
  return 1 - levenshtein(la, lb) / Math.max(la.length, lb.length);
}

function hasEdge(fromId: string, toId: string): boolean {
  const [a, b] = fromId < toId ? [fromId, toId] : [toId, fromId];
  return !!getDb().prepare("SELECT 1 FROM edges WHERE from_id = ? AND to_id = ?").get(a, b);
}

function rewireEdges(fromId: string, toId: string): void {
  const db = getDb();
  const edges = db.prepare("SELECT * FROM edges WHERE from_id = ? OR to_id = ?").all(fromId, fromId) as any[];
  for (const e of edges) {
    const src = e.from_id === fromId ? toId : e.from_id;
    const dst = e.to_id   === fromId ? toId : e.to_id;
    if (src === dst) continue;
    const [a, b] = src < dst ? [src, dst] : [dst, src];
    if (!getDb().prepare("SELECT 1 FROM edges WHERE from_id = ? AND to_id = ?").get(a, b)) {
      db.prepare(`INSERT INTO edges (from_id,to_id,weight,type,co_occurrences,created_at,last_reinforced_at)
                  VALUES (?,?,?,?,?,?,?)`)
        .run(a, b, e.weight, e.type, e.co_occurrences, e.created_at, e.last_reinforced_at);
    }
  }
  db.prepare("DELETE FROM edges WHERE from_id = ? OR to_id = ?").run(fromId, fromId);
}

export interface MaintenanceReport {
  skipped?: boolean;
  decay: ReturnType<typeof consolidate> | null;
  semanticLinksAdded: number;
  newLinks: Array<{ a: string; b: string; similarity: number }>;
  autoMerged: number;
  merges: Array<{ kept: string; deleted: string; similarity: number }>;
  orphansPruned: number;
  conceptualHubsCreated: number;
  hubs: string[];
  ranAt: string;
  durationMs: number;
}

export function runMaintenance(force = false): MaintenanceReport {
  const start = Date.now();
  const report: MaintenanceReport = {
    decay: null,
    semanticLinksAdded: 0,
    newLinks: [],
    autoMerged: 0,
    merges: [],
    orphansPruned: 0,
    conceptualHubsCreated: 0,
    hubs: [],
    ranAt: new Date().toISOString(),
    durationMs: 0,
  };

  // Rate-limit: skip if run recently (unless forced)
  if (!force) {
    const last = getMeta("last_maintenance");
    if (last) {
      const elapsedHours = (Date.now() - parseInt(last)) / (1000 * 60 * 60);
      if (elapsedHours < MIN_INTERVAL_HOURS) {
        return { ...report, skipped: true, durationMs: Date.now() - start };
      }
    }
  }

  // ── Step 1: Ebbinghaus decay ────────────────────────────────────────────
  report.decay = consolidate();

  // ── Step 2: Semantic linking & Synonym detection ────────────────────────
  // After decay, reload nodes (some may have been deleted)
  const nodes = getAllNodes().filter(n => n.embedding !== null);

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      if (hasEdge(a.id, b.id)) continue;

      const sim = cosineSimilarity(a.embedding!, b.embedding!);
      if (sim < SEMANTIC_LINK_THRESHOLD) continue;

      const aIsCtx = a.label.startsWith("[ctx:");
      const bIsCtx = b.label.startsWith("[ctx:");
      if (aIsCtx !== bIsCtx) continue; 

      // Language-agnostic detection:
      // If embedding similarity is extremely high (> 0.94) but text similarity is low,
      // it's likely a translation or a deep synonym.
      const textSim = textSimilarity(a.label, b.label);
      const isSynonym = sim >= 0.94 && textSim < 0.3;

      const weight = isSynonym
        ? 0.9 // Stronger link for cross-language synonyms
        : (sim - SEMANTIC_LINK_THRESHOLD) / (1 - SEMANTIC_LINK_THRESHOLD) * 0.4;

      const type = isSynonym ? "semantic" : "semantic"; // Keeping type simple for now but weight conveys strength
      upsertEdge(a.id, b.id, type, weight);
      
      report.semanticLinksAdded++;
      if (report.newLinks.length < 10) {
        report.newLinks.push({ a: a.label, b: b.label, similarity: Math.round(sim * 100) / 100 });
      }
    }
  }

  // ── Step 3: Auto-merge near-duplicates (Language Agnostic) ──────────────
  const nodesForMerge = getAllNodes().filter(n => !n.label.startsWith("[ctx:"));

  for (let i = 0; i < nodesForMerge.length; i++) {
    for (let j = i + 1; j < nodesForMerge.length; j++) {
      const a = nodesForMerge[i], b = nodesForMerge[j];
      const la = a.label.toLowerCase().trim();
      const lb = b.label.toLowerCase().trim();

      const exactMatch = la === lb;
      const textSim = textSimilarity(a.label, b.label);
      const textMatch = textSim >= 0.92;

      const embSim = (a.embedding && b.embedding)
        ? cosineSimilarity(a.embedding, b.embedding)
        : 0;

      // Human-like merge: relax text requirements if embeddings are near-identical.
      // This helps merge "pablo" and "Pablo" but also "Cat" and "Gato" if the model supports it.
      // For now, we only merge if text is somewhat similar OR embeddings are SUPER high.
      const embMatch = embSim >= 0.985; // Extremely high confidence
      const mixedMatch = embSim >= 0.96 && textSim >= 0.4; // Strong confidence + some lexical hint

      if (!exactMatch && !textMatch && !embMatch && !mixedMatch) continue;

      // Keep the one with more history or higher importance
      const aRank = a.access_count * (1 + a.importance);
      const bRank = b.access_count * (1 + b.importance);
      const [keep, del] = aRank >= bRank ? [a, b] : [b, a];
      
      rewireEdges(del.id, keep.id);
      deleteNode(del.id);

      report.autoMerged++;
      if (report.merges.length < 10) {
        report.merges.push({ kept: keep.label, deleted: del.label, similarity: Math.round(embSim * 100) / 100 } as any);
      }

      nodesForMerge.splice(j, 1);
      j--;
    }
  }

  // ── Step 4: Orphan pruning ──────────────────────────────────────────────
  const now = Date.now();
  const afterMerge = getAllNodes();
  const allEdges = getAllEdges();
  const connectedIds = new Set(allEdges.flatMap(e => [e.from_id, e.to_id]));

  for (const n of afterMerge) {
    if (connectedIds.has(n.id)) continue;
    // Don't prune important or strong nodes, even if isolated
    if (n.importance > 0.7 || n.strength > 0.4) continue;
    
    if (n.strength > ORPHAN_MAX_STRENGTH) continue;
    const ageDays = (now - n.last_accessed_at) / (1000 * 60 * 60 * 24);
    if (ageDays < ORPHAN_MIN_AGE_DAYS) continue;
    deleteNode(n.id);
    report.orphansPruned++;
  }

  // ── Build Adjacency Map for Analysis ────────────────────────────────────
  const allEdgesForAnalysis = getAllEdges();
  const adjacency = new Map<string, string[]>();
  for (const e of allEdgesForAnalysis) {
    if (!adjacency.has(e.from_id)) adjacency.set(e.from_id, []);
    if (!adjacency.has(e.to_id))   adjacency.set(e.to_id,   []);
    adjacency.get(e.from_id)!.push(e.to_id);
    adjacency.get(e.to_id)!.push(e.from_id);
  }

  // ── Step 5: Centrality-based Importance Boost (Human focus) ─────────────
  const { updateNodeImportance } = require("./graph");
  const nodesAfterPrune = getAllNodes();

  for (const node of nodesAfterPrune) {
    const neighbors = adjacency.get(node.id) ?? [];
    if (neighbors.length >= 3) {
      // Human focus: things connected to many things become more important automatically.
      const centralityBoost = Math.min(0.2, neighbors.length * 0.02);
      const newImportance = Math.min(1.0, (node.importance || 0.5) + centralityBoost);
      if (newImportance > (node.importance || 0.5) + 0.01) {
        updateNodeImportance(node.id, newImportance);
      }
    }
  }

  // ── Step 6: Conceptual Abstraction (LTP & Chunking) ─────────────────────
  // Detect dense clusters of nodes and create a "concept" node representing them.
  const coreNodes = nodesAfterPrune.filter(n => !n.label.startsWith("[") && !n.label.startsWith("concept:"));
  const processedForChunk = new Set<string>();

  for (const node of coreNodes) {
    if (processedForChunk.has(node.id)) continue;
    
    const neighbors = (adjacency.get(node.id) ?? [])
      .map(id => nodesAfterPrune.find(n => n.id === id))
      .filter((n): n is NonNullable<typeof n> => !!n && !n.label.startsWith("concept:"));

    if (neighbors.length >= 3) {
      // Potential cluster: check internal density (clique-ish)
      const cluster = [node, ...neighbors];
      const clusterIds = new Set(cluster.map(c => c.id));
      
      let internalEdges = 0;
      for (const id of clusterIds) {
        for (const neighborId of (adjacency.get(id) ?? [])) {
          if (clusterIds.has(neighborId)) internalEdges++;
        }
      }
      internalEdges /= 2;

      const possibleEdges = (cluster.length * (cluster.length - 1)) / 2;
      // If cluster is > 60% dense, it's a concept.
      if (internalEdges / possibleEdges >= 0.60) {
        const sorted = cluster.sort((a, b) => (b.importance || 0) - (a.importance || 0));
        const hubLabel = `concept:${sorted[0].label} & others`;
        
        const { findOrCreateNode: createNode } = require("./graph");
        const hub = createNode(hubLabel, null, 0.7); 
        
        for (const member of cluster) {
          upsertEdge(hub.id, member.id, "abstraction", 0.6);
          processedForChunk.add(member.id);
        }
        
        report.conceptualHubsCreated++;
        report.hubs.push(hubLabel);
      }
    }
  }

  setMeta("last_maintenance", String(now));
  report.durationMs = Date.now() - start;
  return report;
}

// ── CLI entry point ─────────────────────────────────────────────────────────
if (require.main === module) {
  const force = process.argv.includes("--force");
  const report = runMaintenance(force);

  if (report.skipped) {
    process.stdout.write("Skipped (ran recently). Use --force to override.\n");
    process.exit(0);
  }

  process.stdout.write(JSON.stringify(report, null, 2) + "\n");
  process.stdout.write(`\n✓ Maintenance complete in ${report.durationMs}ms\n`);
  process.stdout.write(`  Decay: ${report.decay?.nodesDecayed ?? 0} nodes decayed, ${report.decay?.nodesDeleted ?? 0} deleted\n`);
  process.stdout.write(`  Linked: ${report.semanticLinksAdded} new semantic edges\n`);
  process.stdout.write(`  Merged: ${report.autoMerged} near-duplicate nodes\n`);
  process.stdout.write(`  Pruned: ${report.orphansPruned} orphan nodes\n`);
  process.stdout.write(`  Chunked: ${report.conceptualHubsCreated} conceptual hubs created\n`);
}
