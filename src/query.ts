/**
 * UserPromptSubmit hook — injects personal memory context before every response.
 *
 * Key design principle: the injected context must be authoritative enough that
 * Claude uses it directly, without needing to fall back to web search or ask
 * "which QuickList do you mean?". Each entry is formatted as a named knowledge
 * card with type, description, and relationships.
 */

import { getAllNodes, getAllEdges } from "./graph";
import { getEmbedding, findSimilar } from "./embeddings";
import { spreadActivation } from "./activation";

const TOP_K = 5;
const SIM_THRESHOLD = 0.30;
const MIN_RESULTS = 1;

async function main() {
  let raw = "";
  for await (const chunk of process.stdin) raw += chunk;

  let prompt = "";
  try {
    const input = JSON.parse(raw);
    prompt = input.message ?? input.prompt ?? "";
  } catch {
    process.exit(0);
  }

  if (!prompt || prompt.length < 3) process.exit(0);

  try {
    const embedding = await getEmbedding(prompt);
    const allNodes = getAllNodes();
    // Scoring parameters
    const SIM_WEIGHT = 0.50;
    const ACT_WEIGHT = 0.35;
    const STR_WEIGHT = 0.15;

    const similar = findSimilar(embedding, allNodes, SIM_THRESHOLD, TOP_K * 4);

    if (similar.length < MIN_RESULTS) process.exit(0);

    // Human-like refinement: Identify "hidden" associations
    // If a node has high semantic similarity but low lexical overlap, it might be 
    // a translation or a deep conceptual link. We value these highly.
    const promptLower = prompt.toLowerCase();
    const refinedSimilar = similar.map(s => {
      const node = allNodes.find(n => n.id === s.id);
      if (!node) return s;

      const labelLower = node.label.toLowerCase();
      const hasLexicalOverlap = labelLower.split(/\s+/).some(w => w.length > 2 && promptLower.includes(w));
      
      // If high similarity (>0.6) but NO lexical overlap, it's a likely translation or conceptual link.
      // We give it a "Cognitive Insight" boost.
      const insightBoost = (!hasLexicalOverlap && s.similarity > 0.6) ? 0.15 : 0;
      
      // Small boost for exact substring matches (still useful for names/technical terms)
      const lexicalBoost = hasLexicalOverlap ? 0.1 : 0;

      return { ...s, similarity: Math.min(1.0, s.similarity + insightBoost + lexicalBoost) };
    }).sort((a, b) => b.similarity - a.similarity);

    const seedIds = refinedSimilar.slice(0, 5).map(s => ({ id: s.id, activation: 1.0 }));
    const result = await spreadActivation(seedIds, 2, 0.45, 0.08);

    const simMap = new Map(refinedSimilar.map(s => [s.id, s.similarity]));
    const baseSimMap = new Map(similar.map(s => [s.id, s.similarity]));

    // Filter and rank
    const CODE_FILE_RE = /\.(ts|js|tsx|jsx|py|php|rb|go|rs|sh|md|json)$/i;

    const ranked = result.nodes
      .filter(n => !n.label.startsWith("[ctx:"))
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

    if (ranked.length === 0) process.exit(0);

    // Build adjacency for descriptions
    const nodeMap = new Map(allNodes.map(n => [n.id, n.label]));
    const allEdges = getAllEdges();
    const adjacency = new Map<string, Array<{ id: string; weight: number }>>();
    for (const e of allEdges) {
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

      let card = `▸ ${n.label}`;
      if (types.length   > 0) card += `\n  is/of: ${types.join(", ")}`;
      if (related.length > 0) card += `\n  related: ${related.join(", ")}`;
      return card;
    });

    const context = [
      `[Memory Context — Associative recall optimized for current intent. Direct usage recommended.]`,
      ``,
      ...cards,
    ].join("\n");

    process.stdout.write(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: context,
      },
    }));
  } catch {
    process.exit(0);
  }
}

main();
