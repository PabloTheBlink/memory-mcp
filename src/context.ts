import { execSync } from "child_process";
import path from "path";
import os from "os";
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

/**
 * Get a unique identifier for the current device (MAC address)
 */
export function getDeviceId(): string {
  const interfaces = os.networkInterfaces();
  for (const name in interfaces) {
    const ifaces = interfaces[name];
    if (!ifaces) continue;
    for (const iface of ifaces) {
      if (!iface.internal && iface.mac !== '00:00:00:00:00:00') {
        return iface.mac;
      }
    }
  }
  return 'unknown-device';
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

export async function getActiveContext(): Promise<string> {
  const deviceId = getDeviceId();
  const context = await getMeta(`active_context:${deviceId}`);
  if (context) return context;
  
  // Fallback to legacy global meta or detection
  return await getMeta("active_context") ?? detectContext();
}

export async function setActiveContext(context: string): Promise<void> {
  const deviceId = getDeviceId();
  await setMeta(`active_context:${deviceId}`, context);
  // Also update legacy for backward compatibility if needed, 
  // but we prefer device-specific now.
  await setMeta("active_context", context);
}

export async function ensureContextNode(contextName: string): Promise<string> {
  const label = contextLabel(contextName);
  let node = await findOrCreateNode(label);

  if (!node.embedding) {
    // Embed a human-readable description, not the label syntax
    const description = contextName.startsWith("project:")
      ? `software project ${contextName.replace("project:", "")}`
      : contextName;
    const emb = await getEmbedding(description);
    await updateNodeEmbedding(node.id, emb);
  }

  await touchNode(node.id);
  return node.id;
}

// Bind a memory node to the active context with a weak episodic edge.
// Weight is inversely proportional to the node's existing connections to this context
// (first encounter = stronger encoding, like novelty-driven LTP in the hippocampus).
export async function bindToContext(nodeId: string, contextNodeId: string): Promise<void> {
  await upsertEdge(nodeId, contextNodeId, "episodic", 0.05);
}

