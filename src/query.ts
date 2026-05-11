/**
 * UserPromptSubmit hook — injects personal memory context before every response.
 *
 * Key design principle: the injected context must be authoritative enough that
 * Claude uses it directly, without needing to fall back to web search or ask
 * "which QuickList do you mean?". Each entry is formatted as a named knowledge
 * card with type, description, and relationships.
 */

import { getAllNodes, getAllEdges } from "./graph";
import { getEmbedding, findSimilar, cosineSimilarity } from "./embeddings";
import { spreadActivation } from "./activation";

const TOP_K = 5;
const SIM_THRESHOLD = 0.30;
const MIN_RESULTS = 1;

export async function queryMemory(prompt: string): Promise<string | null> {
  if (!prompt || prompt.length < 3) return null;

  try {
    const embedding = await getEmbedding(prompt);
    const allNodes = getAllNodes();
    // Scoring parameters
    const SIM_WEIGHT = 0.50;
    const ACT_WEIGHT = 0.35;
    const STR_WEIGHT = 0.15;

    const similar = findSimilar(embedding, allNodes, SIM_THRESHOLD, TOP_K * 4);

    if (similar.length < MIN_RESULTS) return null;

    // Human-like refinement: Identify "hidden" associations (Language Agnostic)
    const STOP_WORDS = new Set(["the", "and", "for", "with", "that", "this", "los", "las", "con", "para", "que", "una", "uno", "del", "por", "des", "les", "une", "est"]);
    const promptLower = prompt.toLowerCase();
    const promptTokens = promptLower.split(/[\s,.;:!?]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));

    const refinedSimilar = similar.map(s => {
      const node = allNodes.find(n => n.id === s.id);
      if (!node) return s;

      const labelLower = node.label.toLowerCase();
      const labelTokens = labelLower.split(/[\s,.;:!?]+/).filter(t => t.length > 2 && !STOP_WORDS.has(t));
      
      const hasLexicalOverlap = labelTokens.some(t => promptTokens.includes(t)) || 
                               (labelLower.length > 3 && promptLower.includes(labelLower));
      
      // If high similarity (>0.65) but NO lexical overlap, it's a "Cognitive Insight" (likely translation)
      const insightBoost = (!hasLexicalOverlap && s.similarity > 0.65) ? 0.20 : 0;
      
      // Small boost for exact matches
      const lexicalBoost = hasLexicalOverlap ? 0.08 : 0;

      // Importance factor: human memory prioritizes important concepts in search
      const importanceBoost = (node.importance || 0.5) * 0.1;

      return { ...s, similarity: Math.min(1.0, s.similarity + insightBoost + lexicalBoost + importanceBoost) };
    }).sort((a, b) => b.similarity - a.similarity);

    const seedIds = refinedSimilar.slice(0, 6).map(s => ({ id: s.id, activation: 1.0 }));
    const result = await spreadActivation(seedIds, 3, 0.48, 0.06);

    const simMap = new Map(refinedSimilar.map(s => [s.id, s.similarity]));
    const baseSimMap = new Map(similar.map(s => [s.id, s.similarity]));

    // Filter and rank
    const CODE_FILE_RE = /\.(ts|js|tsx|jsx|py|php|rb|go|rs|sh|md|json)$/i;

    const ranked = result.nodes
      .filter(n => !n.label.startsWith("[ctx:") && !n.label.startsWith("rule:") && !n.label.startsWith("preference:"))
      .filter(n => {
        const sim = simMap.get(n.id) ?? 0;
        const baseSim = baseSimMap.get(n.id) ?? 0;
        
        // Noise reduction for code files and technical nodes
        if (CODE_FILE_RE.test(n.label)) {
          return baseSim > 0.75 || promptLower.includes(n.label.toLowerCase());
        }

        // General relevance threshold
        return sim > 0.35 || n.activation > 0.4;
      })
      .map(n => ({
        id: n.id,
        label: n.label,
        score: (simMap.get(n.id) ?? 0) * SIM_WEIGHT + 
               n.activation * ACT_WEIGHT + 
               n.strength * STR_WEIGHT,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    if (ranked.length === 0) return null;

    // Build adjacency for descriptions
    const nodeMap = new Map(allNodes.map(n => [n.id, n.label]));
    const allEdgesData = getAllEdges();
    const adjacency = new Map<string, Array<{ id: string; weight: number }>>();
    for (const e of allEdgesData) {
      if (!adjacency.has(e.from_id)) adjacency.set(e.from_id, []);
      if (!adjacency.has(e.to_id))   adjacency.set(e.to_id,   []);
      adjacency.get(e.from_id)!.push({ id: e.to_id,   weight: e.weight });
      adjacency.get(e.to_id)!.push(  { id: e.from_id, weight: e.weight });
    }

    // Render each node as a knowledge card
    const cards = ranked.map(n => {
      const neighbors = (adjacency.get(n.id) ?? [])
        .filter(nb => {
          const lbl = nodeMap.get(nb.id);
          return lbl && !lbl.startsWith("[ctx:");
        })
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 5)
        .map(nb => ({ label: nodeMap.get(nb.id)!, weight: nb.weight }));

      // High-weight neighbors (≥0.45) read as "type" descriptors or strong associations
      const types   = neighbors.filter(nb => nb.weight >= 0.45).map(nb => nb.label);
      // Medium-weight are contextual relationships
      const related = neighbors.filter(nb => nb.weight >= 0.2 && nb.weight < 0.45).map(nb => nb.label);

      // Check for conceptual hubs (abstraction edges)
      const hubEdges = allEdgesData.filter(e => (e.from_id === n.id || e.to_id === n.id) && e.type === "abstraction");
      const hubs = hubEdges.map(e => {
        const hubId = e.from_id === n.id ? e.to_id : e.from_id;
        return nodeMap.get(hubId);
      }).filter(h => h && h.startsWith("concept:")).map(h => h!.replace("concept:", ""));

      let card = `▸ ${n.label}`;
      if (hubs.length    > 0) card += `\n  part of concept: ${hubs.join(", ")}`;
      if (types.length   > 0) card += `\n  is/of: ${types.join(", ")}`;
      if (related.length > 0) card += `\n  related: ${related.join(", ")}`;
      return card;
    });

    // Fetch active rules relevant to the prompt
    const activeRules = allNodes
      .filter(n => n.label.startsWith("rule:") || n.label.startsWith("preference:"))
      .map(n => ({ label: n.label, sim: n.embedding ? cosineSimilarity(embedding, n.embedding) : 0 }))
      .filter(r => r.sim > 0.35)
      .sort((a, b) => b.sim - a.sim)
      .slice(0, 3)
      .map(r => r.label.replace(/^(rule:|preference:)/, "• "));

    const contextParts = [
      `[Memory Context — Associative recall optimized for current intent. Direct usage recommended.]`,
      `[SYSTEM DIRECTIVE]: You are an autonomous agent with a persistent memory. ALWAYS use memory_recall at the start of a task to fetch user preferences and context. ALWAYS use memory_learn_rule or memory_activate to persist new patterns, style preferences, or technical insights. This reduces token waste and improves alignment.`,
    ];

    if (activeRules.length > 0) {
      contextParts.push(`\n[Active Rules & Preferences]:\n${activeRules.join("\n")}`);
    }

    return [
      ...contextParts,
      ``,
      ...cards,
    ].join("\n");
  } catch (err) {
    return null;
  }
}

async function main() {
  if (require.main !== module) return;

  let raw = "";
  for await (const chunk of process.stdin) raw += chunk;

  let prompt = "";
  try {
    const input = JSON.parse(raw);
    prompt = input.message ?? input.prompt ?? "";
  } catch {
    process.exit(0);
  }

  const context = await queryMemory(prompt);
  if (!context) process.exit(0);

  process.stdout.write(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: context,
    },
  }));
}

main();
