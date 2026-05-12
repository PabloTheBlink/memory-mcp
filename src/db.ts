import Database from "better-sqlite3";
import mysql from "mysql2/promise";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";

// Robust dotenv loading using the directory of the current file
dotenv.config({ path: path.join(__dirname, "../.env") });
dotenv.config();

const DB_TYPE = (process.env.DB_TYPE || "").trim() === "mysql" ? "mysql" : "sqlite";
process.stderr.write(`[DB] Using adapter: ${DB_TYPE}\n`);
if (DB_TYPE === "mysql") {
  process.stderr.write(`[DB] MySQL Host: ${process.env.MYSQL_HOST}\n`);
}

let DB_PATH = path.join(__dirname, "../data/memory.db");

export interface DBAdapter {
  initSchema(): Promise<void>;
  close(): Promise<void>;
  queryAll(sql: string, params?: any[]): Promise<any[]>;
  queryGet(sql: string, params?: any[]): Promise<any>;
  run(sql: string, params?: any[]): Promise<void>;
  setMeta(key: string, value: string): Promise<void>;
}

class SqliteAdapter implements DBAdapter {
  private db: Database.Database | null = null;

  constructor() {
    fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
    this.db = new Database(DB_PATH);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
  }

  async initSchema(): Promise<void> {
    if (!this.db) return;
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        embedding TEXT,
        strength REAL NOT NULL DEFAULT 0.5,
        importance REAL NOT NULL DEFAULT 0.5,
        access_count INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL,
        last_accessed_at INTEGER NOT NULL,
        last_fired_at INTEGER NOT NULL DEFAULT 0,
        metadata TEXT,
        user_id TEXT,
        visibility TEXT DEFAULT 'private'
      );

      CREATE TABLE IF NOT EXISTS edges (
        from_id TEXT NOT NULL,
        to_id TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 0.5,
        type TEXT NOT NULL DEFAULT 'semantic',
        co_occurrences INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL,
        last_reinforced_at INTEGER NOT NULL,
        user_id TEXT,
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
      CREATE INDEX IF NOT EXISTS idx_nodes_user ON nodes(user_id);
    `);

    const info = this.db.prepare("PRAGMA table_info(nodes)").all() as any[];
    if (!info.some(col => col.name === "metadata")) {
      this.db.exec("ALTER TABLE nodes ADD COLUMN metadata TEXT");
    }
    if (!info.some(col => col.name === "user_id")) {
      this.db.exec("ALTER TABLE nodes ADD COLUMN user_id TEXT");
    }
    if (!info.some(col => col.name === "visibility")) {
      this.db.exec("ALTER TABLE nodes ADD COLUMN visibility TEXT DEFAULT 'private'");
    }

    const edgeInfo = this.db.prepare("PRAGMA table_info(edges)").all() as any[];
    if (!edgeInfo.some(col => col.name === "user_id")) {
      this.db.exec("ALTER TABLE edges ADD COLUMN user_id TEXT");
    }
    
    // Ensure UNIQUE(label, user_id) - SQLite workaround: 
    // We can't easily change constraints, but we can check if the old unique index exists and drop it if we want to allow same labels for different users.
    // For now, let's keep it simple and just ensure the columns exist.
  }

  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }

  async queryAll(sql: string, params: any[] = []): Promise<any[]> {
    return this.db!.prepare(sql).all(...params);
  }

  async queryGet(sql: string, params: any[] = []): Promise<any> {
    return this.db!.prepare(sql).get(...params);
  }

  async run(sql: string, params: any[] = []): Promise<void> {
    this.db!.prepare(sql).run(...params);
  }

  async setMeta(key: string, value: string): Promise<void> {
    this.db!.prepare("INSERT OR REPLACE INTO meta (`key`, value) VALUES (?, ?)").run(key, value);
  }
}

class MysqlAdapter implements DBAdapter {
  private pool: mysql.Pool;

  constructor() {
    this.pool = mysql.createPool({
      host: process.env.MYSQL_HOST || "localhost",
      port: Number(process.env.MYSQL_PORT) || 3306,
      user: process.env.MYSQL_USER || "root",
      password: process.env.MYSQL_PASSWORD || "",
      database: process.env.MYSQL_DATABASE || "memory_mcp",
      waitForConnections: true,
      connectionLimit: 10,
      queueLimit: 0
    });
  }

  async initSchema(): Promise<void> {
    const conn = await this.pool.getConnection();
    try {
      await conn.query(`
        CREATE TABLE IF NOT EXISTS nodes (
          id VARCHAR(36) PRIMARY KEY,
          label VARCHAR(255) NOT NULL,
          embedding LONGTEXT,
          strength DOUBLE NOT NULL DEFAULT 0.5,
          importance DOUBLE NOT NULL DEFAULT 0.5,
          access_count INT NOT NULL DEFAULT 0,
          created_at BIGINT NOT NULL,
          last_accessed_at BIGINT NOT NULL,
          last_fired_at BIGINT NOT NULL DEFAULT 0,
          metadata LONGTEXT,
          user_id VARCHAR(100),
          visibility VARCHAR(20) DEFAULT 'private'
        );
      `);

      await conn.query(`
        CREATE TABLE IF NOT EXISTS edges (
          from_id VARCHAR(36) NOT NULL,
          to_id VARCHAR(36) NOT NULL,
          weight DOUBLE NOT NULL DEFAULT 0.5,
          type VARCHAR(50) NOT NULL DEFAULT 'semantic',
          co_occurrences INT NOT NULL DEFAULT 1,
          created_at BIGINT NOT NULL,
          last_reinforced_at BIGINT NOT NULL,
          user_id VARCHAR(100),
          PRIMARY KEY (from_id, to_id),
          FOREIGN KEY (from_id) REFERENCES nodes(id) ON DELETE CASCADE,
          FOREIGN KEY (to_id) REFERENCES nodes(id) ON DELETE CASCADE
        );
      `);

      await conn.query(`
        CREATE TABLE IF NOT EXISTS meta (
          \`key\` VARCHAR(255) PRIMARY KEY,
          value LONGTEXT NOT NULL
        );
      `);

      const addColumn = async (table: string, col: string, type: string) => {
        const [cols] = await conn.query<any>(`SHOW COLUMNS FROM ?? LIKE ?`, [table, col]);
        if (cols.length === 0) {
          await conn.query(`ALTER TABLE ?? ADD COLUMN ?? ${type}`, [table, col]);
        }
      };

      await addColumn("nodes", "metadata", "LONGTEXT");
      await addColumn("nodes", "user_id", "VARCHAR(100)");
      await addColumn("nodes", "visibility", "VARCHAR(20) DEFAULT 'private'");
      await addColumn("edges", "user_id", "VARCHAR(100)");

      // Create indexes if they don't exist
      const createIndexIfNotExists = async (tableName: string, indexName: string, columnName: string) => {
        const [rows] = await conn.query<any>(`SHOW INDEX FROM ?? WHERE Key_name = ?`, [tableName, indexName]);
        if (rows.length === 0) {
          await conn.query(`CREATE INDEX ?? ON ??(??)`, [indexName, tableName, columnName]);
        }
      };

      await createIndexIfNotExists("edges", "idx_edges_from", "from_id");
      await createIndexIfNotExists("edges", "idx_edges_to", "to_id");
      await createIndexIfNotExists("nodes", "idx_nodes_strength", "strength");
      await createIndexIfNotExists("nodes", "idx_nodes_label", "label");
      await createIndexIfNotExists("nodes", "idx_nodes_user", "user_id");
    } finally {
      conn.release();
    }
  }

  async close(): Promise<void> {
    await this.pool.end();
  }

  private convertSql(sql: string): string {
    let converted = sql
      .replace(/INSERT OR REPLACE INTO/g, "REPLACE INTO")
      .replace(/INSERT INTO meta \(`key`, value\) VALUES \(\?, \?\) ON DUPLICATE KEY UPDATE value = VALUES\(value\)/g, "INSERT INTO meta (`key`, value) VALUES (?, ?) ON DUPLICATE KEY UPDATE value = VALUES(value)") 
      .replace(/ON CONFLICT\((.*?)\) DO UPDATE SET/g, "ON DUPLICATE KEY UPDATE")
      .replace(/MIN\((.*?,.*?)\)/gi, "LEAST($1)")
      .replace(/MAX\((.*?,.*?)\)/gi, "GREATEST($1)");
    
    return converted;
  }

  async queryAll(sql: string, params: any[] = []): Promise<any[]> {
    const [rows] = await this.pool.query(this.convertSql(sql), params);
    return rows as any[];
  }

  async queryGet(sql: string, params: any[] = []): Promise<any> {
    const [rows] = await this.pool.query(this.convertSql(sql), params);
    const arr = rows as any[];
    return arr.length > 0 ? arr[0] : undefined;
  }

  async run(sql: string, params: any[] = []): Promise<void> {
    try {
      await this.pool.execute(this.convertSql(sql), params);
    } catch (e: any) {
      process.stderr.write(`[DB ERROR] SQL: ${sql}\n`);
      process.stderr.write(`[DB ERROR] Params: ${JSON.stringify(params)}\n`);
      throw e;
    }
  }

  async setMeta(key: string, value: string): Promise<void> {
    await this.pool.execute(
      "INSERT INTO meta (`key`, value) VALUES (?, ?) ON DUPLICATE KEY UPDATE value = VALUES(value)",
      [key, value]
    );
  }
}

let _adapter: DBAdapter | null = null;

export async function getDb(): Promise<DBAdapter> {
  if (_adapter) return _adapter;

  if (DB_TYPE === "mysql") {
    _adapter = new MysqlAdapter();
  } else {
    _adapter = new SqliteAdapter();
  }
  await _adapter.initSchema();
  return _adapter;
}

export async function setDbPath(newPath: string): Promise<void> {
  DB_PATH = newPath;
  if (_adapter) {
    await _adapter.close();
    _adapter = null;
  }
}

export async function closeDb(): Promise<void> {
  if (_adapter) {
    await _adapter.close();
    _adapter = null;
  }
}
