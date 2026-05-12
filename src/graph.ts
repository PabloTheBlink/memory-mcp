import { v4 as uuidv4 } from "uuid";
import { getDb, setDbPath, closeDb, DBAdapter } from "./db";

export { getDb, setDbPath, closeDb };

export interface MemoryNode {
  id: string;
  label: string;
  embedding: number[] | null;
  strength: number;
  importance: number;
  access_count: number;
  created_at: number;
  last_accessed_at: number;
  last_fired_at: number;
  metadata: Record<string, any> | null;
  user_id: string | null;
  visibility: "private" | "shared";
}

export interface MemoryEdge {
  from_id: string;
  to_id: string;
  weight: number;
  type: "causal" | "temporal" | "semantic" | "episodic" | "abstraction";
  co_occurrences: number;
  created_at: number;
  last_reinforced_at: number;
  user_id: string | null;
}

export interface ActivatedNode extends MemoryNode {
  activation: number;
}

function deserializeNode(row: any): MemoryNode {
  return {
    id: row.id,
    label: row.label,
    embedding: row.embedding ? JSON.parse(row.embedding) : null,
    strength: row.strength,
    importance: row.importance,
    access_count: row.access_count,
    created_at: row.created_at,
    last_accessed_at: row.last_accessed_at,
    last_fired_at: row.last_fired_at || 0,
    metadata: row.metadata ? JSON.parse(row.metadata) : null,
    user_id: row.user_id,
    visibility: row.visibility || "private",
  };
}

function deserializeEdge(row: any): MemoryEdge {
  return {
    from_id: row.from_id,
    to_id: row.to_id,
    weight: row.weight,
    type: row.type,
    co_occurrences: row.co_occurrences,
    created_at: row.edge_created_at || row.created_at,
    last_reinforced_at: row.last_reinforced_at,
    user_id: row.user_id,
  };
}

export async function findOrCreateNode(
  label: string,
  embedding: number[] | null = null,
  importance: number = 0.5,
  metadata: Record<string, any> | null = null,
  userId: string | null = null,
  visibility: "private" | "shared" = "private"
): Promise<MemoryNode> {
  label = label.length > 255 ? label.slice(0, 254) + '…' : label;
  const db = await getDb();
  const now = Date.now();

  // Search for the node by label. If label uniqueness is enforced, we must find it regardless of visibility/user
  let existing = await db.queryGet(
    "SELECT * FROM nodes WHERE label = ?", 
    [label]
  );
  
  if (existing) {
    // If it exists but was private for someone else and we want to use it, 
    // we could potentially update its visibility or just use it.
    // For context nodes, they should ideally be shared.
    return deserializeNode(existing);
  }

  const id = uuidv4();
  await db.run(`
    INSERT INTO nodes (id, label, embedding, strength, importance, access_count, created_at, last_accessed_at, last_fired_at, metadata, user_id, visibility)
    VALUES (?, ?, ?, 0.5, ?, 0, ?, ?, 0, ?, ?, ?)
  `, [id, label, embedding ? JSON.stringify(embedding) : null, importance, now, now, metadata ? JSON.stringify(metadata) : null, userId, visibility]);

  return { id, label, embedding, strength: 0.5, importance, access_count: 0, created_at: now, last_accessed_at: now, last_fired_at: 0, metadata, user_id: userId, visibility };
}

export async function getNodeById(id: string): Promise<MemoryNode | null> {
  const db = await getDb();
  const row = await db.queryGet("SELECT * FROM nodes WHERE id = ?", [id]);
  return row ? deserializeNode(row) : null;
}

export async function getAllNodes(userId?: string): Promise<MemoryNode[]> {
  const db = await getDb();
  let sql = "SELECT * FROM nodes";
  let params: any[] = [];
  
  if (userId) {
    sql += " WHERE user_id = ? OR visibility = 'shared'";
    params.push(userId);
  }
  
  const rows = await db.queryAll(sql, params);
  return rows.map(deserializeNode);
}

export async function updateNodeEmbedding(id: string, embedding: number[]): Promise<void> {
  const db = await getDb();
  await db.run("UPDATE nodes SET embedding = ? WHERE id = ?", [JSON.stringify(embedding), id]);
}

export async function updateNodeImportance(id: string, importance: number): Promise<void> {
  const db = await getDb();
  await db.run("UPDATE nodes SET importance = ? WHERE id = ?", [importance, id]);
}

export async function updateNodeStrength(id: string, strength: number): Promise<void> {
  const db = await getDb();
  await db.run("UPDATE nodes SET strength = ? WHERE id = ?", [strength, id]);
}

export async function touchNode(id: string): Promise<void> {
  const db = await getDb();
  const now = Date.now();
  await db.run("UPDATE nodes SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?", [now, id]);
}

export async function fireNode(id: string): Promise<void> {
  const db = await getDb();
  const now = Date.now();
  await db.run("UPDATE nodes SET last_fired_at = ? WHERE id = ?", [now, id]);
}

export async function deleteNode(id: string): Promise<void> {
  const db = await getDb();
  await db.run("DELETE FROM nodes WHERE id = ?", [id]);
}

export async function upsertEdge(
  fromId: string,
  toId: string,
  type: string,
  strengthBoost: number = 0.1,
  userId: string | null = null
): Promise<MemoryEdge> {
  const db = await getDb();
  const now = Date.now();
  const DB_TYPE = (process.env.DB_TYPE || "").trim() === "mysql" ? "mysql" : "sqlite";
  
  const [a, b] = fromId < toId ? [fromId, toId] : [toId, fromId];

  if (DB_TYPE === "mysql") {
    await db.run(`
      INSERT INTO edges (from_id, to_id, type, weight, created_at, last_reinforced_at, co_occurrences, user_id)
      VALUES (?, ?, ?, ?, ?, ?, 1, ?)
      ON DUPLICATE KEY UPDATE
        weight = LEAST(1.0, weight + ?),
        last_reinforced_at = ?,
        co_occurrences = co_occurrences + 1
    `, [a, b, type, strengthBoost, now, now, userId, strengthBoost, now]);
  } else {
    await db.run(`
      INSERT INTO edges (from_id, to_id, type, weight, created_at, last_reinforced_at, co_occurrences, user_id)
      VALUES (?, ?, ?, ?, ?, ?, 1, ?)
      ON CONFLICT(from_id, to_id) DO UPDATE SET
        weight = MIN(1.0, weight + ?),
        last_reinforced_at = ?,
        co_occurrences = co_occurrences + 1
    `, [a, b, type, strengthBoost, now, now, userId, strengthBoost, now]);
  }

  return { from_id: a, to_id: b, weight: 0.5, type: type as any, co_occurrences: 1, created_at: now, last_reinforced_at: now, user_id: userId };
}

export async function updateEdgeWeight(fromId: string, toId: string, weight: number): Promise<void> {
  const db = await getDb();
  const [a, b] = fromId < toId ? [fromId, toId] : [toId, fromId];
  await db.run("UPDATE edges SET weight = ? WHERE from_id = ? AND to_id = ?", [weight, a, b]);
}

export async function deleteEdge(fromId: string, toId: string): Promise<void> {
  const db = await getDb();
  const [a, b] = fromId < toId ? [fromId, toId] : [toId, fromId];
  await db.run("DELETE FROM edges WHERE from_id = ? AND to_id = ?", [a, b]);
}

export async function getAllEdges(userId?: string): Promise<MemoryEdge[]> {
  const db = await getDb();
  let sql = "SELECT * FROM edges";
  let params: any[] = [];
  
  if (userId) {
    // Return edges where both nodes are visible to the user
    // or edges created by the user
    sql = `
      SELECT e.* FROM edges e
      JOIN nodes n1 ON e.from_id = n1.id
      JOIN nodes n2 ON e.to_id = n2.id
      WHERE (n1.user_id = ? OR n1.visibility = 'shared')
      AND (n2.user_id = ? OR n2.visibility = 'shared')
    `;
    params = [userId, userId];
  }
  
  const rows = await db.queryAll(sql, params);
  return rows.map(deserializeEdge);
}

export async function getNeighbors(nodeId: string, userId?: string): Promise<Array<{ node: MemoryNode; edge: MemoryEdge }>> {
  const db = await getDb();
  let sql = `
    SELECT n.*, e.weight, e.type, e.co_occurrences, e.created_at as edge_created_at,
           e.last_reinforced_at, e.from_id, e.to_id, e.user_id as edge_user_id
    FROM edges e
    JOIN nodes n ON (e.to_id = n.id OR e.from_id = n.id)
    WHERE (e.from_id = ? OR e.to_id = ?) AND n.id != ?
  `;
  const params: any[] = [nodeId, nodeId, nodeId];
  
  if (userId) {
    sql += " AND (n.user_id = ? OR n.visibility = 'shared')";
    params.push(userId);
  }
  
  const rows = await db.queryAll(sql, params);

  return rows.map((r: any) => ({
    node: deserializeNode(r),
    edge: deserializeEdge({ ...r, user_id: r.edge_user_id })
  }));
}

export async function getTopNodes(limit: number = 50, userId?: string): Promise<MemoryNode[]> {
  const db = await getDb();
  let sql = "SELECT * FROM nodes";
  const params: any[] = [];
  
  if (userId) {
    sql += " WHERE user_id = ? OR visibility = 'shared'";
    params.push(userId);
  }
  
  sql += " ORDER BY (importance * 0.7 + strength * 0.3) DESC LIMIT ?";
  params.push(limit);
  
  const rows = await db.queryAll(sql, params);
  return rows.map(deserializeNode);
}

export async function getStats(userId?: string): Promise<{ nodes: number; edges: number; nodeCount: number; edgeCount: number }> {
  const db = await getDb();
  let nodeSql = "SELECT COUNT(*) as c FROM nodes";
  let edgeSql = "SELECT COUNT(*) as c FROM edges";
  const params: any[] = [];
  
  if (userId) {
    nodeSql += " WHERE user_id = ? OR visibility = 'shared'";
    edgeSql = `
      SELECT COUNT(*) as c FROM edges e
      JOIN nodes n1 ON e.from_id = n1.id
      JOIN nodes n2 ON e.to_id = n2.id
      WHERE (n1.user_id = ? OR n1.visibility = 'shared')
      AND (n2.user_id = ? OR n2.visibility = 'shared')
    `;
    params.push(userId, userId);
  }
  
  const nodes = await db.queryGet(nodeSql, userId ? [userId] : []);
  const edges = await db.queryGet(edgeSql, params);
  return { 
    nodes: nodes.c, 
    edges: edges.c,
    nodeCount: nodes.c,
    edgeCount: edges.c
  };
}

export async function getMeta(key: string): Promise<string | null> {
  const db = await getDb();
  const row = await db.queryGet("SELECT value FROM meta WHERE `key` = ?", [key]);
  return row ? row.value : null;
}

export async function setMeta(key: string, value: string): Promise<void> {
  const db = await getDb();
  await db.setMeta(key, value);
}

