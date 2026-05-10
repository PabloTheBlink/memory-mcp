import Database from "better-sqlite3";
import { v4 as uuidv4 } from "uuid";
import path from "path";
import fs from "fs";

const DB_PATH = path.join(__dirname, "../data/memory.db");

export interface MemoryNode {
  id: string;
  label: string;
  embedding: number[] | null;
  strength: number;
  importance: number; // 0-1, how significant this memory is
  access_count: number;
  created_at: number;
  last_accessed_at: number;
  last_fired_at: number;
  metadata: Record<string, any> | null;
}

export interface MemoryEdge {
  from_id: string;
  to_id: string;
  weight: number;
  type: "causal" | "temporal" | "semantic" | "episodic" | "abstraction";
  co_occurrences: number;
  created_at: number;
  last_reinforced_at: number;
}

export interface ActivatedNode extends MemoryNode {
  activation: number;
}

let _db: Database.Database | null = null;

export function getDb(): Database.Database {
  if (_db) return _db;

  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
  _db = new Database(DB_PATH);
  _db.pragma("journal_mode = WAL");
  _db.pragma("foreign_keys = ON");
  initSchema(_db);
  return _db;
}

function initSchema(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS nodes (
      id TEXT PRIMARY KEY,
      label TEXT NOT NULL UNIQUE,
      embedding TEXT,
      strength REAL NOT NULL DEFAULT 0.5,
      importance REAL NOT NULL DEFAULT 0.5,
      access_count INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL,
      last_accessed_at INTEGER NOT NULL,
      last_fired_at INTEGER NOT NULL DEFAULT 0,
      metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS edges (
      from_id TEXT NOT NULL,
      to_id TEXT NOT NULL,
      weight REAL NOT NULL DEFAULT 0.5,
      type TEXT NOT NULL DEFAULT 'semantic',
      co_occurrences INTEGER NOT NULL DEFAULT 1,
      created_at INTEGER NOT NULL,
      last_reinforced_at INTEGER NOT NULL,
      PRIMARY KEY (from_id, to_id),
      FOREIGN KEY (from_id) REFERENCES nodes(id) ON DELETE CASCADE,
      FOREIGN KEY (to_id) REFERENCES nodes(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
    CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
    CREATE INDEX IF NOT EXISTS idx_nodes_strength ON nodes(strength);
    CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
  `);

  // Migration: Add metadata column if it doesn't exist
  const info = db.prepare("PRAGMA table_info(nodes)").all() as any[];
  if (!info.some(col => col.name === "metadata")) {
    db.exec("ALTER TABLE nodes ADD COLUMN metadata TEXT");
  }
}

export function findOrCreateNode(
  label: string, 
  embedding: number[] | null = null, 
  importance: number = 0.5,
  metadata: Record<string, any> | null = null
): MemoryNode {
  const db = getDb();
  const now = Date.now();

  const existing = db.prepare("SELECT * FROM nodes WHERE label = ?").get(label) as any;
  if (existing) {
    return deserializeNode(existing);
  }

  const id = uuidv4();
  db.prepare(`
    INSERT INTO nodes (id, label, embedding, strength, importance, access_count, created_at, last_accessed_at, last_fired_at, metadata)
    VALUES (?, ?, ?, 0.5, ?, 0, ?, ?, 0, ?)
  `).run(id, label, embedding ? JSON.stringify(embedding) : null, importance, now, now, metadata ? JSON.stringify(metadata) : null);

  return { id, label, embedding, strength: 0.5, importance, access_count: 0, created_at: now, last_accessed_at: now, last_fired_at: 0, metadata };
}

export function getNodeById(id: string): MemoryNode | null {
  const db = getDb();
  const row = db.prepare("SELECT * FROM nodes WHERE id = ?").get(id) as any;
  return row ? deserializeNode(row) : null;
}

export function getAllNodes(): MemoryNode[] {
  const db = getDb();
  return (db.prepare("SELECT * FROM nodes").all() as any[]).map(deserializeNode);
}

export function updateNodeEmbedding(id: string, embedding: number[]): void {
  getDb().prepare("UPDATE nodes SET embedding = ? WHERE id = ?").run(JSON.stringify(embedding), id);
}

export function touchNode(id: string): void {
  const now = Date.now();
  getDb().prepare(`
    UPDATE nodes SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?
  `).run(now, id);
}

export function fireNode(id: string): void {
  const now = Date.now();
  getDb().prepare(`
    UPDATE nodes SET last_fired_at = ? WHERE id = ?
  `).run(now, id);
}

export function getNeighbors(nodeId: string): Array<{ node: MemoryNode; edge: MemoryEdge }> {
  const db = getDb();
  const rows = db.prepare(`
    SELECT n.*, e.weight, e.type, e.co_occurrences, e.created_at as edge_created_at,
           e.last_reinforced_at, e.from_id, e.to_id
    FROM edges e
    JOIN nodes n ON (e.to_id = n.id OR e.from_id = n.id)
    WHERE (e.from_id = ? OR e.to_id = ?) AND n.id != ?
  `).all(nodeId, nodeId, nodeId) as any[];

  return rows.map((r) => ({
    node: deserializeNode(r),
    edge: {
      from_id: r.from_id,
      to_id: r.to_id,
      weight: r.weight,
      type: r.type,
      co_occurrences: r.co_occurrences,
      created_at: r.edge_created_at,
      last_reinforced_at: r.last_reinforced_at,
    },
  }));
}

export function upsertEdge(
  fromId: string,
  toId: string,
  type: string,
  strengthBoost: number = 0.1
): MemoryEdge {
  const db = getDb();
  const now = Date.now();
  const [a, b] = fromId < toId ? [fromId, toId] : [toId, fromId];

  const existing = db.prepare("SELECT * FROM edges WHERE from_id = ? AND to_id = ?").get(a, b) as any;

  if (existing) {
    const newWeight = Math.min(1.0, existing.weight + strengthBoost);
    db.prepare(`
      UPDATE edges SET weight = ?, co_occurrences = co_occurrences + 1, last_reinforced_at = ?
      WHERE from_id = ? AND to_id = ?
    `).run(newWeight, now, a, b);
    return deserializeEdge({ ...existing, weight: newWeight, co_occurrences: existing.co_occurrences + 1, last_reinforced_at: now });
  }

  const initialWeight = Math.min(1.0, 0.3 + strengthBoost);
  db.prepare(`
    INSERT INTO edges (from_id, to_id, weight, type, co_occurrences, created_at, last_reinforced_at)
    VALUES (?, ?, ?, ?, 1, ?, ?)
  `).run(a, b, initialWeight, type, now, now);

  return { from_id: a, to_id: b, weight: initialWeight, type: type as any, co_occurrences: 1, created_at: now, last_reinforced_at: now };
}

export function getAllEdges(): MemoryEdge[] {
  return (getDb().prepare("SELECT * FROM edges").all() as any[]).map(deserializeEdge);
}

export function deleteNode(id: string): void {
  getDb().prepare("DELETE FROM nodes WHERE id = ?").run(id);
}

export function deleteEdge(fromId: string, toId: string): void {
  const [a, b] = fromId < toId ? [fromId, toId] : [toId, fromId];
  getDb().prepare("DELETE FROM edges WHERE from_id = ? AND to_id = ?").run(a, b);
}

export function updateNodeStrength(id: string, strength: number): void {
  getDb().prepare("UPDATE nodes SET strength = ? WHERE id = ?").run(Math.max(0, Math.min(1, strength)), id);
}

export function updateNodeImportance(id: string, importance: number): void {
  getDb().prepare("UPDATE nodes SET importance = ? WHERE id = ?").run(Math.max(0, Math.min(1, importance)), id);
}

export function updateEdgeWeight(fromId: string, toId: string, weight: number): void {
  const [a, b] = fromId < toId ? [fromId, toId] : [toId, fromId];
  getDb().prepare("UPDATE edges SET weight = ? WHERE from_id = ? AND to_id = ?").run(Math.max(0, Math.min(1, weight)), a, b);
}

export function getTopNodes(limit: number): MemoryNode[] {
  return (getDb().prepare("SELECT * FROM nodes ORDER BY strength DESC LIMIT ?").all(limit) as any[]).map(deserializeNode);
}

export function getStats(): { nodeCount: number; edgeCount: number; lastConsolidation: number | null } {
  const db = getDb();
  const nodeCount = (db.prepare("SELECT COUNT(*) as c FROM nodes").get() as any).c;
  const edgeCount = (db.prepare("SELECT COUNT(*) as c FROM edges").get() as any).c;
  const meta = db.prepare("SELECT value FROM meta WHERE key = 'last_consolidation'").get() as any;
  return {
    nodeCount,
    edgeCount,
    lastConsolidation: meta ? parseInt(meta.value) : null,
  };
}

export function setMeta(key: string, value: string): void {
  getDb().prepare("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)").run(key, value);
}

export function getMeta(key: string): string | null {
  const row = getDb().prepare("SELECT value FROM meta WHERE key = ?").get(key) as any;
  return row ? row.value : null;
}

function deserializeNode(row: any): MemoryNode {
  return {
    id: row.id,
    label: row.label,
    embedding: row.embedding ? JSON.parse(row.embedding) : null,
    strength: row.strength,
    importance: row.importance ?? 0.5,
    access_count: row.access_count,
    created_at: row.created_at,
    last_accessed_at: row.last_accessed_at,
    last_fired_at: row.last_fired_at ?? 0,
    metadata: row.metadata ? JSON.parse(row.metadata) : null,
  };
}

function deserializeEdge(row: any): MemoryEdge {
  return {
    from_id: row.from_id,
    to_id: row.to_id,
    weight: row.weight,
    type: row.type,
    co_occurrences: row.co_occurrences,
    created_at: row.created_at,
    last_reinforced_at: row.last_reinforced_at,
  };
}
