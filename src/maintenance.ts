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

import { 
  getAllNodes, 
  getAllEdges, 
  getDb, 
  upsertEdge, 
  deleteNode, 
  setMeta, 
  getMeta, 
  rewireEdges,
  updateNodeImportance,
  findOrCreateNode as createNode,
  updateNodeEmbedding
} from "./graph";
import { cosineSimilarity, getEmbedding } from "./embeddings";
import { consolidate } from "./decay";

const SEMANTIC_LINK_THRESHOLD  = 0.72;  // create edge if no edge exists (lowered from 0.78)
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
  const now = start; 
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
  process.stderr.write("[Maintenance] Step 1: Decay...\n");
  report.decay = await consolidate();

  // Load all nodes and edges into memory for efficient O(N^2) processing
  let allNodes = await getAllNodes();
  let allEdges = await getAllEdges();
  process.stderr.write(`[Maintenance] Loaded ${allNodes.length} nodes and ${allEdges.length} edges.\n`);
  
  // Helper for efficient edge lookup
  const getEdgeKey = (a: string, b: string) => a < b ? `${a}:${b}` : `${b}:${a}`;
  const edgeSet = new Set(allEdges.map(e => getEdgeKey(e.from_id, e.to_id)));

  // ── Step 2: Semantic linking & Synonym detection ────────────────────────
  process.stderr.write("[Maintenance] Step 2: Semantic linking...\n");
  const nodesForLinking = allNodes.filter(n => n.embedding !== null);

  for (let i = 0; i < nodesForLinking.length; i++) {
    if (i % 20 === 0) process.stderr.write(`  Linking progress: ${i}/${nodesForLinking.length}\r`);
    const a = nodesForLinking[i];
    
    for (let j = i + 1; j < nodesForLinking.length; j++) {
      const b = nodesForLinking[j];
      
      // Only link nodes from the same user or if both are shared
      if (a.user_id !== b.user_id && (a.visibility !== 'shared' || b.visibility !== 'shared')) continue;
      
      if (edgeSet.has(getEdgeKey(a.id, b.id))) continue;

      // Coarse filter for performance
      let coarseScore = 0;
      for (let k = 0; k < 8; k++) {
        coarseScore += a.embedding![k] * b.embedding![k];
      }
      if (coarseScore <= 0) continue;

      const sim = cosineSimilarity(a.embedding!, b.embedding!);
      if (sim < SEMANTIC_LINK_THRESHOLD) continue;

      const aIsCtx = a.label.startsWith("[ctx:");
      const bIsCtx = b.label.startsWith("[ctx:");
      if (aIsCtx !== bIsCtx) continue; 

      const textSim = textSimilarity(a.label, b.label);
      const isSynonym = sim >= 0.94 && textSim < 0.3;

      const weight = isSynonym
        ? 0.9 
        : (sim - SEMANTIC_LINK_THRESHOLD) / (1 - SEMANTIC_LINK_THRESHOLD) * 0.4;

      await upsertEdge(a.id, b.id, "semantic", weight, a.user_id);
      edgeSet.add(getEdgeKey(a.id, b.id)); // Update local cache
      
      report.semanticLinksAdded++;
      if (report.newLinks.length < 10) {
        report.newLinks.push({ a: a.label, b: b.label, similarity: Math.round(sim * 100) / 100 });
      }
    }
  }
  process.stderr.write(`\n  Semantic linking complete: ${report.semanticLinksAdded} links added.\n`);

  // ── Step 3: Auto-merge near-duplicates ──────────────────────────────────
  process.stderr.write("[Maintenance] Step 3: Auto-merge...\n");
  allNodes = await getAllNodes(); 
  const nodesForMerge = allNodes.filter(n => !n.label.startsWith("[ctx:") && !n.label.startsWith("rule:") && !n.label.startsWith("preference:"));
  const deletedIds = new Set<string>();

  for (let i = 0; i < nodesForMerge.length; i++) {
    if (i % 20 === 0) process.stderr.write(`  Merge progress: ${i}/${nodesForMerge.length}\r`);
    const a = nodesForMerge[i];
    if (deletedIds.has(a.id)) continue;

    for (let j = i + 1; j < nodesForMerge.length; j++) {
      const b = nodesForMerge[j];
      if (deletedIds.has(b.id)) continue;
      
      if (a.user_id !== b.user_id) continue;
      
      const la = a.label.toLowerCase().trim();
      const lb = b.label.toLowerCase().trim();

      const textSim = textSimilarity(a.label, b.label);
      
      // Coarse filter
      let coarseScore = 0;
      if (a.embedding && b.embedding) {
        for (let k = 0; k < 8; k++) {
          coarseScore += a.embedding[k] * b.embedding[k];
        }
      }
      if (coarseScore <= 0 && textSim < 0.8) continue;

      const embSim = (a.embedding && b.embedding) ? cosineSimilarity(a.embedding, b.embedding) : 0;

      const exactMatch = la === lb;
      const textMatch = textSim >= 0.92;
      const embMatch = embSim >= 0.94; 
      const mixedMatch = embSim >= 0.88 && textSim >= 0.5; 

      if (!exactMatch && !textMatch && !embMatch && !mixedMatch) continue;

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

      if (del === a) break;

    }
  }
  process.stderr.write(`\n  Auto-merge complete: ${report.autoMerged} nodes merged.\n`);

  // ── Step 4: Islands Auto-Linking ─────────────────────────────────────────
  process.stderr.write("[Maintenance] Step 4: Islands & Weakly Connected...\n");
  const ISLAND_LINK_THRESHOLD = 0.58; // Lowered from 0.65
  const nodesAfterMerge = await getAllNodes();
  allEdges = await getAllEdges();
  
  const neighborCounts = new Map<string, number>();
  for (const e of allEdges) {
    neighborCounts.set(e.from_id, (neighborCounts.get(e.from_id) || 0) + 1);
    neighborCounts.set(e.to_id, (neighborCounts.get(e.to_id) || 0) + 1);
  }

  for (const node of nodesAfterMerge) {
    const count = neighborCounts.get(node.id) || 0;
    // Target nodes that are islands (0) or weakly connected (1)
    if (count >= 2) continue;
    if (node.label.startsWith("[ctx:") || node.label.startsWith("curiosity:") || !node.embedding) continue;

    const matches: Array<{ target: typeof node; sim: number }> = [];

    for (const target of nodesAfterMerge) {
      if (target.id === node.id || !target.embedding) continue;
      
      const sim = cosineSimilarity(node.embedding, target.embedding);
      if (sim >= ISLAND_LINK_THRESHOLD) {
        matches.push({ target, sim });
      }
    }

    // Link to top 3 matches to build a more robust graph
    const topMatches = matches.sort((a, b) => b.sim - a.sim).slice(0, 3);
    for (const match of topMatches) {
      await upsertEdge(node.id, match.target.id, "semantic", 0.35, node.user_id);
      report.islandsLinked++;
    }
  }

  // ── Step 5: Strict Orphan Pruning ─────────────────────────────────────────
  process.stderr.write("[Maintenance] Step 5: Pruning...\n");
  const finalEdges = await getAllEdges();
  const finalConnectedIds = new Set(finalEdges.flatMap(e => [e.from_id, e.to_id]));
  
  for (const n of nodesAfterMerge) {
    if (finalConnectedIds.has(n.id)) continue;
    // Don't prune rules, preferences, or context hubs
    if (n.label.startsWith("rule:") || n.label.startsWith("preference:") || n.label.startsWith("[ctx:")) continue;
    // Don't prune important nodes
    if (n.importance >= 0.8) continue;

    await deleteNode(n.id);
    report.orphansPruned++;
  }

  // ── Step 6: Centrality-based Importance Boost ─────────────────────────────
  process.stderr.write("[Maintenance] Step 6: Centrality...\n");
  const nodesAfterPrune = await getAllNodes();
  const edgesAfterPrune = await getAllEdges();
  const adjacency = new Map<string, string[]>();
  for (const e of edgesAfterPrune) {
    if (!adjacency.has(e.from_id)) adjacency.set(e.from_id, []);
    if (!adjacency.has(e.to_id))   adjacency.set(e.to_id,   []);
    adjacency.get(e.from_id)!.push(e.to_id);
    adjacency.get(e.to_id)!.push(e.from_id);
  }

  for (const node of nodesAfterPrune) {
    const neighbors = adjacency.get(node.id) ?? [];
    if (neighbors.length >= 3) {
      const centralityBoost = Math.min(0.2, neighbors.length * 0.02);
      const newImportance = Math.min(1.0, (node.importance || 0.5) + centralityBoost);
      if (newImportance > (node.importance || 0.5) + 0.01) {
        await updateNodeImportance(node.id, newImportance);
      }
    }
  }

  // ── Step 7: Conceptual Abstraction ───────────────────────────────────────
  process.stderr.write("[Maintenance] Step 7: Abstraction...\n");
  const nodesMap = new Map(nodesAfterPrune.map(n => [n.id, n]));
  const coreNodes = nodesAfterPrune.filter(n => !n.label.startsWith("[") && !n.label.startsWith("concept:"));
  const processedForChunk = new Set<string>();

  for (const node of coreNodes) {
    if (processedForChunk.has(node.id)) continue;
    
    const neighbors = (adjacency.get(node.id) ?? [])
      .map(id => nodesMap.get(id))
      .filter((n): n is NonNullable<typeof n> => !!n && !n.label.startsWith("concept:"));

    if (neighbors.length >= 3) {
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
      if (internalEdges / possibleEdges >= 0.60) {
        const sorted = cluster.sort((a, b) => (b.importance || 0) - (a.importance || 0));
        const hubLabel = `concept:${sorted[0].label} & others`;
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

  // ── Step 8: Conflict Detection ───────────────────────────────────────────
  process.stderr.write("[Maintenance] Step 8: Conflict Detection...\n");
  const NEGATION_WORDS = new Set(["no", "never", "not", "nunca", "jamas", "jamás", "neither", "nor", "tampoco"]);
  const nodesForConflict = nodesAfterPrune.filter(n => !n.label.startsWith("["));

  for (let i = 0; i < nodesForConflict.length; i++) {
    for (let j = i + 1; j < nodesForConflict.length; j++) {
      const a = nodesForConflict[i], b = nodesForConflict[j];
      if (!a.embedding || !b.embedding) continue;

      const sim = cosineSimilarity(a.embedding, b.embedding);
      if (sim < 0.82) continue;

      const tokensA = a.label.toLowerCase().split(/\s+/);
      const tokensB = b.label.toLowerCase().split(/\s+/);
      
      const hasNegA = tokensA.some(t => NEGATION_WORDS.has(t));
      const hasNegB = tokensB.some(t => NEGATION_WORDS.has(t));

      if (hasNegA !== hasNegB) {
        const conflictLabel = `conflict:${a.label} VS ${b.label}`;
        const conflictNode = await createNode(conflictLabel, null, 0.9, null, a.user_id, a.visibility); 
        
        await upsertEdge(conflictNode.id, a.id, "causal", 0.5, a.user_id);
        await upsertEdge(conflictNode.id, b.id, "causal", 0.5, b.user_id);
        
        report.hubs.push(conflictLabel);
      }
    }
  }

  // ── Step 9: Curiosity Engine ─────────────────────────────────────────────
  process.stderr.write("[Maintenance] Step 9: Curiosity...\n");
  for (const node of nodesAfterPrune) {
    if (node.label.startsWith("[") || node.label.startsWith("concept:") || node.label.startsWith("conflict:")) continue;
    
    const neighbors = adjacency.get(node.id) ?? [];
    if (node.importance > 0.7 && neighbors.length < 2) {
      const baseLabel = node.label.length > 180 ? node.label.slice(0, 180) + '…' : node.label;
      const curiosityLabel = `curiosity:Tell me more about "${baseLabel}" to bridge knowledge gaps`;
      const curiosityNode = await createNode(curiosityLabel, null, 0.4, null, node.user_id, node.visibility);
      
      await upsertEdge(curiosityNode.id, node.id, "semantic", 0.3, node.user_id);
      report.hubs.push(curiosityLabel);
    }
  }

  // ── Step 9.5: Batch Embedding Generation for New Hubs ─────────────────────
  process.stderr.write("[Maintenance] Step 9.5: Hub Embeddings...\n");
  const hubsWithoutEmbeddings = (await getAllNodes()).filter(n => n.embedding === null && (n.label.startsWith("concept:") || n.label.startsWith("conflict:") || n.label.startsWith("curiosity:")));
  if (hubsWithoutEmbeddings.length > 0) {
    const labels = hubsWithoutEmbeddings.map(h => h.label);
    const { getEmbeddings } = require("./embeddings");
    const embeddings = await getEmbeddings(labels);
    for (let i = 0; i < hubsWithoutEmbeddings.length; i++) {
      await updateNodeEmbedding(hubsWithoutEmbeddings[i].id, embeddings[i]);
    }
  }

  // ── Step 10: Importance Re-calibration ───────────────────────────────────
  process.stderr.write("[Maintenance] Step 10: Re-calibration...\n");
  for (const node of nodesAfterPrune) {
    const lifespanDays = (now - node.created_at) / (1000 * 60 * 60 * 24);
    if (lifespanDays > 1 && node.access_count > 5) {
      const utilityScore = Math.min(0.4, (node.access_count / lifespanDays) * 0.1);
      const newImportance = Math.min(1.0, (node.importance || 0.5) + utilityScore);
      if (newImportance > (node.importance || 0.5) + 0.05) {
        await updateNodeImportance(node.id, newImportance);
      }
    }
  }

  await setMeta("last_maintenance", String(now));
  report.durationMs = Date.now() - start;
  process.stderr.write(`[Maintenance] Complete in ${report.durationMs}ms.\n`);
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
