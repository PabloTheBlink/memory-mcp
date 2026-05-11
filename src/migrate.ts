import Database from "better-sqlite3";
import mysql from "mysql2/promise";
import path from "path";
import dotenv from "dotenv";
import fs from "fs";
import { getDb } from "./db";

dotenv.config();

async function migrate() {
  console.log("Starting migration from SQLite to MySQL...");
  
  const sqlitePath = path.join(__dirname, "../data/memory.db");
  if (!fs.existsSync(sqlitePath)) {
    console.error("SQLite database not found at", sqlitePath);
    process.exit(1);
  }
  
  const sqliteDb = new Database(sqlitePath, { fileMustExist: true });

  const pool = mysql.createPool({
    host: process.env.MYSQL_HOST || "localhost",
    port: Number(process.env.MYSQL_PORT) || 3306,
    user: process.env.MYSQL_USER || "root",
    password: process.env.MYSQL_PASSWORD || "",
    database: process.env.MYSQL_DATABASE || "memory_mcp",
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
  });

  const conn = await pool.getConnection();

  try {
    console.log("Initializing MySQL schema...");
    // Ensure the DB_TYPE points to mysql so getDb() uses MySQL logic to init schema
    process.env.DB_TYPE = "mysql";
    await getDb(); // This will initialize schema
    console.log("Schema initialized.");

    // Nodes
    const nodes = sqliteDb.prepare("SELECT * FROM nodes").all() as any[];
    console.log(`Migrating ${nodes.length} nodes...`);
    for (let i = 0; i < nodes.length; i += 100) {
      const chunk = nodes.slice(i, i + 100);
      const values = chunk.map(n => [
        n.id, n.label, n.embedding, n.strength, n.importance, 
        n.access_count, n.created_at, n.last_accessed_at, n.last_fired_at, n.metadata
      ]);
      await conn.query(
        "INSERT IGNORE INTO nodes (id, label, embedding, strength, importance, access_count, created_at, last_accessed_at, last_fired_at, metadata) VALUES ?",
        [values]
      );
    }

    // Edges
    const edges = sqliteDb.prepare("SELECT * FROM edges").all() as any[];
    console.log(`Migrating ${edges.length} edges...`);
    for (let i = 0; i < edges.length; i += 100) {
      const chunk = edges.slice(i, i + 100);
      const values = chunk.map(e => [
        e.from_id, e.to_id, e.weight, e.type, e.co_occurrences, e.created_at, e.last_reinforced_at
      ]);
      await conn.query(
        "INSERT IGNORE INTO edges (from_id, to_id, weight, type, co_occurrences, created_at, last_reinforced_at) VALUES ?",
        [values]
      );
    }

    // Meta
    const metas = sqliteDb.prepare("SELECT * FROM meta").all() as any[];
    console.log(`Migrating ${metas.length} meta records...`);
    for (const m of metas) {
      await conn.query(
        "INSERT INTO meta (`key`, value) VALUES (?, ?) ON DUPLICATE KEY UPDATE value = VALUES(value)",
        [m.key, m.value]
      );
    }

    console.log("Migration completed successfully!");
  } catch (err) {
    console.error("Migration failed:", err);
  } finally {
    conn.release();
    await pool.end();
    sqliteDb.close();
  }
}

migrate();
