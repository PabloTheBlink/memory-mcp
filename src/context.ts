import { execSync } from "child_process";
import path from "path";
import { findOrCreateNode, getNodeById, getMeta, setMeta, upsertEdge, updateNodeEmbedding, touchNode } from "./graph";
import { getEmbedding } from "./embeddings";

// Context nodes use a reserved label prefix so they're identifiable
// but otherwise live as regular nodes in the graph — just very well connected hubs.
const CTX_PREFIX = "[ctx:";

export function contextLabel(name: string): string {
  return `${CTX_PREFIX}${name}]`;
}

export function isContextNode(label: string): boolean {
  return label.startsWith(CTX_PREFIX);
}

// Detect active context from git repo name or fall back to cwd basename
export function detectContext(): string {
  try {
    const remote = execSync("git remote get-url origin 2>/dev/null", { encoding: "utf8" }).trim();
    if (remote) {
      const match = remote.match(/\/([^/]+?)(\.git)?$/);
      if (match) return `project:${match[1]}`;
    }
  } catch {}

  try {
    const toplevel = execSync("git rev-parse --show-toplevel 2>/dev/null", { encoding: "utf8" }).trim();
    if (toplevel) return `project:${path.basename(toplevel)}`;
  } catch {}

  return `project:${path.basename(process.cwd())}`;
}

export function getActiveContext(): string {
  return getMeta("active_context") ?? detectContext();
}

export async function ensureContextNode(contextName: string): Promise<string> {
  const label = contextLabel(contextName);
  let node = findOrCreateNode(label);

  if (!node.embedding) {
    // Embed a human-readable description, not the label syntax
    const description = contextName.startsWith("project:")
      ? `software project ${contextName.replace("project:", "")}`
      : contextName;
    const emb = await getEmbedding(description);
    updateNodeEmbedding(node.id, emb);
  }

  touchNode(node.id);
  return node.id;
}

// Bind a memory node to the active context with a weak episodic edge.
// Weight is inversely proportional to the node's existing connections to this context
// (first encounter = stronger encoding, like novelty-driven LTP in the hippocampus).
export function bindToContext(nodeId: string, contextNodeId: string): void {
  upsertEdge(nodeId, contextNodeId, "episodic", 0.05);
}
