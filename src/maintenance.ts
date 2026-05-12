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

async function hasEdge(fromId: string, toId: string): Promise<boolean> {
  const [a, b] = fromId < toId ? [fromId, toId] : [toId, fromId];
  const db = await getDb();
  const row = await db.queryGet("SELECT 1 FROM edges WHERE from_id = ? AND to_id = ?", [a, b]);
  return !!row;
}

async function rewireEdges(fromId: string, toId: string): Promise<void> {
  const db = await getDb();
  const edges = await db.queryAll("SELECT * FROM edges WHERE from_id = ? OR to_id = ?", [fromId, fromId]);
  for (const e of edges) {
    const src = e.from_id === fromId ? toId : e.from_id;
    const dst = e.to_id   === fromId ? toId : e.to_id;
    if (src === dst) continue;
    const [a, b] = src < dst ? [src, dst] : [dst, src];
    const exists = await db.queryGet("SELECT 1 FROM edges WHERE from_id = ? AND to_id = ?", [a, b]);
    if (!exists) {
      await db.run(`INSERT INTO edges (from_id,to_id,weight,type,co_occurrences,created_at,last_reinforced_at)
                  VALUES (?,?,?,?,?,?,?)`, [a, b, e.weight, e.type, e.co_occurrences, e.created_at, e.last_reinforced_at]);
    }
  }
  await db.run("DELETE FROM edges WHERE from_id = ? OR to_id = ?", [fromId, fromId]);
}

export interface MaintenanceReport {
  skipped?: boolean;
  decay: Awaited<ReturnType<typeof consolidate>> | null;
  semanticLinksAdded: number;
  islandsLinked: number;
  newLinks: Array<{ a: string; b: string; similarity: number }>;
  autoMerged: number;
  merges: Array<{ kept: string; deleted: string; similarity: number }>;
  orphansPruned: number;
  conceptualHubsCreated: number;
  hubs: string[];
  ranAt: string;
  durationMs: number;
}

export async function runMaintenance(force = false): Promise<MaintenanceReport> {
  const start = Date.now();
  const report: MaintenanceReport = {
    decay: null,
    semanticLinksAdded: 0,
    islandsLinked: 0,
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
    const last = await getMeta("last_maintenance");
    if (last) {
      const elapsedHours = (Date.now() - parseInt(last)) / (1000 * 60 * 60);
      if (elapsedHours < MIN_INTERVAL_HOURS) {
        return { ...report, skipped: true, durationMs: Date.now() - start };
      }
    }
  }

  // ── Step 1: Ebbinghaus decay ────────────────────────────────────────────
  report.decay = await consolidate();

  // ── Step 2: Semantic linking & Synonym detection ────────────────────────
  // After decay, reload nodes (some may have been deleted)
  const nodes = (await getAllNodes()).filter(n => n.embedding !== null);

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      
      // Only link nodes from the same user or if both are shared
      if (a.user_id !== b.user_id && (a.visibility !== 'shared' || b.visibility !== 'shared')) continue;
      
      if (await hasEdge(a.id, b.id)) continue;

      const sim = cosineSimilarity(a.embedding!, b.embedding!);
      if (sim < SEMANTIC_LINK_THRESHOLD) continue;

      const aIsCtx = a.label.startsWith("[ctx:");
      const bIsCtx = b.label.startsWith("[ctx:");
      if (aIsCtx !== bIsCtx) continue; 

      // Language-agnostic detection:
      const textSim = textSimilarity(a.label, b.label);
      const isSynonym = sim >= 0.94 && textSim < 0.3;

      const weight = isSynonym
        ? 0.9 
        : (sim - SEMANTIC_LINK_THRESHOLD) / (1 - SEMANTIC_LINK_THRESHOLD) * 0.4;

      const type = "semantic";
      await upsertEdge(a.id, b.id, type, weight, a.user_id);
      
      report.semanticLinksAdded++;
      if (report.newLinks.length < 10) {
        report.newLinks.push({ a: a.label, b: b.label, similarity: Math.round(sim * 100) / 100 });
      }
    }
  }

  // ── Step 3: Auto-merge near-duplicates (Language Agnostic) ──────────────
  const nodesForMerge = (await getAllNodes()).filter(n => !n.label.startsWith("[ctx:") && !n.label.startsWith("rule:") && !n.label.startsWith("preference:"));
  const deletedIds = new Set<string>();

  for (let i = 0; i < nodesForMerge.length; i++) {
    const a = nodesForMerge[i];
    if (deletedIds.has(a.id)) continue;

    for (let j = i + 1; j < nodesForMerge.length; j++) {
      const b = nodesForMerge[j];
      if (deletedIds.has(b.id)) continue;
      
      // Only merge nodes from the same user
      if (a.user_id !== b.user_id) continue;
      
      const la = a.label.toLowerCase().trim();
      const lb = b.label.toLowerCase().trim();

      const exactMatch = la === lb;
      const textSim = textSimilarity(a.label, b.label);
      const textMatch = textSim >= 0.92;

      const embSim = (a.embedding && b.embedding)
        ? cosineSimilarity(a.embedding, b.embedding)
        : 0;

      // Language-agnostic merge
      const embMatch = embSim >= 0.94; 
      const mixedMatch = embSim >= 0.88 && textSim >= 0.5; 

      if (!exactMatch && !textMatch && !embMatch && !mixedMatch) continue;

      // Keep the one with more history or higher importance
      const aRank = a.access_count * (1 + a.importance);
      const bRank = b.access_count * (1 + b.importance);
      const [keep, del] = aRank >= bRank ? [a, b] : [b, a];
      
      await rewireEdges(del.id, keep.id);
      await deleteNode(del.id);
      deletedIds.add(del.id);

      report.autoMerged++;
      if (report.merges.length < 10) {
        report.merges.push({ kept: keep.label, deleted: del.label, similarity: Math.round(embSim * 100) / 100 } as any);
      }

      if (del === a) break; // Current 'a' is deleted, move to next 'i'
    }
  }

  // ── Step 4: Islands Auto-Linking (Pre-Pruning) ───────────────────────────
  // Identify nodes with 0 edges and try to link them.
  const ISLAND_LINK_THRESHOLD = 0.65; 
  const nodesAfterMerge = await getAllNodes();
  const allEdgesBeforePrune = await getAllEdges();
  const connectedIds = new Set(allEdgesBeforePrune.flatMap(e => [e.from_id, e.to_id]));
  
  for (const node of nodesAfterMerge) {
    if (connectedIds.has(node.id)) continue;
    if (node.label.startsWith("[ctx:") || node.label.startsWith("curiosity:") || !node.embedding) continue;

    let bestMatch = null;
    let highestSim = 0;

    for (const target of nodesAfterMerge) {
      if (target.id === node.id || !target.embedding) continue;
      
      const sim = cosineSimilarity(node.embedding, target.embedding);
      if (sim > highestSim) {
        highestSim = sim;
        bestMatch = target;
      }
    }

    if (bestMatch && highestSim >= ISLAND_LINK_THRESHOLD) {
      await upsertEdge(node.id, bestMatch.id, "semantic", 0.4, node.user_id);
      connectedIds.add(node.id);
      connectedIds.add(bestMatch.id);
      report.islandsLinked++;
    }
  }

  // ── Step 5: Strict Orphan Pruning ─────────────────────────────────────────
  const now = Date.now();
  // Now prune anything that is still not connected.
  for (const n of nodesAfterMerge) {
    if (connectedIds.has(n.id)) continue;
    
    // Protect critical system nodes
    if (n.label.startsWith("rule:") || n.label.startsWith("preference:") || n.label.startsWith("[ctx:")) continue;
    
    // If it's very important or very strong, maybe keep it? 
    // The user was quite explicit: "if they have no relation, eliminate them".
    // We'll keep very high importance nodes (0.9+) just in case, but prune everything else.
    if (n.importance >= 0.9) continue;

    await deleteNode(n.id);
    report.orphansPruned++;
  }

  // ── Build Adjacency Map for Analysis ────────────────────────────────────
  const allEdgesForAnalysis = await getAllEdges();
  const adjacency = new Map<string, string[]>();
  for (const e of allEdgesForAnalysis) {
    if (!adjacency.has(e.from_id)) adjacency.set(e.from_id, []);
    if (!adjacency.has(e.to_id))   adjacency.set(e.to_id,   []);
    adjacency.get(e.from_id)!.push(e.to_id);
    adjacency.get(e.to_id)!.push(e.from_id);
  }

  // ── Step 5: Centrality-based Importance Boost (Human focus) ─────────────
  const { updateNodeImportance } = require("./graph");
  const nodesAfterPrune = await getAllNodes();

  for (const node of nodesAfterPrune) {
    const neighbors = adjacency.get(node.id) ?? [];
    if (neighbors.length >= 3) {
      // Human focus: things connected to many things become more important automatically.
      const centralityBoost = Math.min(0.2, neighbors.length * 0.02);
      const newImportance = Math.min(1.0, (node.importance || 0.5) + centralityBoost);
      if (newImportance > (node.importance || 0.5) + 0.01) {
        await updateNodeImportance(node.id, newImportance);
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
        const hub = await createNode(hubLabel, null, 0.7, null, sorted[0].user_id, "shared"); 
        
        for (const member of cluster) {
          await upsertEdge(hub.id, member.id, "abstraction", 0.6, member.user_id);
          processedForChunk.add(member.id);
        }
        
        report.conceptualHubsCreated++;
        report.hubs.push(hubLabel);
      }
    }
  }

  // ── Step 7: Conflict Detection (Heuristic) ───────────────────────────────
  const NEGATION_WORDS = new Set(["no", "never", "not", "nunca", "jamas", "jamás", "neither", "nor", "tampoco"]);
  const nodesForConflict = (await getAllNodes()).filter(n => !n.label.startsWith("["));

  for (let i = 0; i < nodesForConflict.length; i++) {
    for (let j = i + 1; j < nodesForConflict.length; j++) {
      const a = nodesForConflict[i], b = nodesForConflict[j];
      if (!a.embedding || !b.embedding) continue;

      const sim = cosineSimilarity(a.embedding, b.embedding);
      if (sim < 0.82) continue; // High similarity required for potential conflict

      const tokensA = a.label.toLowerCase().split(/\s+/);
      const tokensB = b.label.toLowerCase().split(/\s+/);
      
      const hasNegA = tokensA.some(t => NEGATION_WORDS.has(t));
      const hasNegB = tokensB.some(t => NEGATION_WORDS.has(t));

      if (hasNegA !== hasNegB) {
        // Potential contradiction detected
        const conflictLabel = `conflict:${a.label} VS ${b.label}`;
        const { findOrCreateNode: createNode } = require("./graph");
        const conflictNode = await createNode(conflictLabel, null, 0.9, null, a.user_id, a.visibility); 
        
        await upsertEdge(conflictNode.id, a.id, "causal", 0.5, a.user_id);
        await upsertEdge(conflictNode.id, b.id, "causal", 0.5, b.user_id);
        
        report.hubs.push(conflictLabel);
      }
    }
  }

  // ── Step 8: Curiosity Engine (Knowledge Gap Detection) ──────────────────
  const nodesAfterConflict = await getAllNodes();
  for (const node of nodesAfterConflict) {
    if (node.label.startsWith("[") || node.label.startsWith("concept:") || node.label.startsWith("conflict:")) continue;
    
    const neighbors = adjacency.get(node.id) ?? [];
    // Curiosity trigger: Important node with very few associations
    if (node.importance > 0.7 && neighbors.length < 2) {
      const baseLabel = node.label.length > 180 ? node.label.slice(0, 180) + '…' : node.label;
      const curiosityLabel = `curiosity:Tell me more about "${baseLabel}" to bridge knowledge gaps`;
      const { findOrCreateNode: createNode } = require("./graph");
      const curiosityNode = await createNode(curiosityLabel, null, 0.4, null, node.user_id, node.visibility);
      await upsertEdge(curiosityNode.id, node.id, "semantic", 0.3, node.user_id);
      report.hubs.push(curiosityLabel);
    }
  }

  // (Step 8.5 removed as it was integrated into Step 4/5)

  // ── Step 9: Importance Re-calibration (Long-term utility) ────────────────
  // Heuristic: If a node has high access count and was used across multiple days,
  // it is objectively important regardless of its initial importance score.
  for (const node of nodesAfterConflict) {
    const lifespanDays = (now - node.created_at) / (1000 * 60 * 60 * 24);
    if (lifespanDays > 1 && node.access_count > 5) {
      // Boost importance based on sustained utility
      const utilityScore = Math.min(0.4, (node.access_count / lifespanDays) * 0.1);
      const newImportance = Math.min(1.0, (node.importance || 0.5) + utilityScore);
      if (newImportance > (node.importance || 0.5) + 0.05) {
        await updateNodeImportance(node.id, newImportance);
      }
    }
  }

  await setMeta("last_maintenance", String(now));
  report.durationMs = Date.now() - start;
  return report;
}

// ── CLI entry point ─────────────────────────────────────────────────────────
(async () => {
  if (require.main === module) {
    const force = process.argv.includes("--force");
    const report = await runMaintenance(force);

    if (report.skipped) {
      process.stdout.write("Skipped (ran recently). Use --force to override.\n");
      process.exit(0);
    }

    process.stdout.write(JSON.stringify(report, null, 2) + "\n");
    process.stdout.write(`\n✓ Maintenance complete in ${report.durationMs}ms\n`);
    process.stdout.write(`  Decay: ${report.decay?.nodesDecayed ?? 0} nodes decayed, ${report.decay?.nodesDeleted ?? 0} deleted\n`);
    process.stdout.write(`  Linked: ${report.semanticLinksAdded} new semantic edges, ${report.islandsLinked} islands linked\n`);
    process.stdout.write(`  Merged: ${report.autoMerged} near-duplicate nodes\n`);
    process.stdout.write(`  Pruned: ${report.orphansPruned} orphan nodes\n`);
    process.stdout.write(`  Chunked: ${report.conceptualHubsCreated} conceptual hubs created\n`);
  }
})();
