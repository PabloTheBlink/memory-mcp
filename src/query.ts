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
    const similar = findSimilar(embedding, allNodes, SIM_THRESHOLD, TOP_K * 3);

    if (similar.length < MIN_RESULTS) process.exit(0);

    // Boost nodes whose label appears as a keyword in the prompt
    const promptLower = prompt.toLowerCase();
    const boostedSimilar = similar.map(s => {
      const node = allNodes.find(n => n.id === s.id);
      const labelLower = node?.label.toLowerCase() ?? "";
      // Keyword match: label word appears in prompt or vice versa
      const words = labelLower.split(/\s+/);
      const boost = words.some(w => w.length > 2 && promptLower.includes(w)) ? 0.3 : 0;
      return { ...s, similarity: s.similarity + boost };
    }).sort((a, b) => b.similarity - a.similarity);

    const seedIds = boostedSimilar.slice(0, 5).map(s => ({ id: s.id, activation: 1.0 }));
    const result = await spreadActivation(seedIds, 2, 0.5, 0.1);

    const simMap = new Map(boostedSimilar.map(s => [s.id, s.similarity]));
    const keywordMatchedIds = new Set(
      boostedSimilar.filter(s => s.similarity > SIM_THRESHOLD + 0.25).map(s => s.id)
    );

    // Code file labels (e.g. planner.ts, loop.ts) are noisy — the embedding model
    // gives them high similarity to any short query. Only include them if they
    // were directly keyword-matched.
    const CODE_FILE_RE = /\.(ts|js|tsx|jsx|py|php|rb|go|rs|sh)$/i;

    const ranked = result.nodes
      .filter(n => !n.label.startsWith("[ctx:"))
      .filter(n => {
        const sim = simMap.get(n.id) ?? 0;
        if (sim <= 0.01) return false;
        // Filter out code-file nodes unless keyword matched
        if (CODE_FILE_RE.test(n.label) && !keywordMatchedIds.has(n.id)) return false;
        // Non-keyword-matched nodes need a reasonably high base similarity
        const baseSimMap = new Map(similar.map(s => [s.id, s.similarity]));
        const baseSim = baseSimMap.get(n.id) ?? 0;
        if (!keywordMatchedIds.has(n.id) && baseSim < 0.42) return false;
        return true;
      })
      .map(n => ({
        id: n.id,
        label: n.label,
        score: (simMap.get(n.id) ?? 0) * 0.6 + n.activation * 0.3 + n.strength * 0.1,
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

      // High-weight neighbors (≥0.5) read as "type" descriptors
      const types   = neighbors.filter(nb => nb.weight >= 0.5).map(nb => nb.label);
      // Medium-weight are contextual relationships
      const related = neighbors.filter(nb => nb.weight >= 0.3 && nb.weight < 0.5).map(nb => nb.label);

      let card = `▸ ${n.label}`;
      if (types.length   > 0) card += `\n  is: ${types.join(", ")}`;
      if (related.length > 0) card += `\n  related: ${related.join(", ")}`;
      return card;
    });

    const context = [
      `[Personal Memory — verified context for this user. Use this directly; do NOT search the web for these concepts.]`,
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
