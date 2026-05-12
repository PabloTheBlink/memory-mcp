import { getDb } from "../src/db";

async function checkCapabilities() {
  const db = await getDb();
  
  console.log("--- DB Capabilities Check ---");
  
  // MySQL check
  if (process.env.DB_TYPE === "mysql") {
    try {
      const version = await db.queryGet("SELECT VERSION() as v");
      console.log(`MySQL Version: ${version.v}`);
      
      // Check if VECTOR type exists (MySQL 8.0.30+)
      try {
        await db.run("CREATE TEMPORARY TABLE vec_test (v VECTOR(384))");
        console.log("Native VECTOR support: YES");
      } catch (e) {
        console.log("Native VECTOR support: NO");
      }
    } catch (e) {
      console.log(`MySQL Error: ${e}`);
    }
  } else {
    // SQLite check
    try {
      const version = await db.queryGet("SELECT sqlite_version() as v");
      console.log(`SQLite Version: ${version.v}`);
      
      try {
        await db.run("SELECT vss_version()");
        console.log("sqlite-vss support: YES");
      } catch (e) {
        console.log("sqlite-vss support: NO");
      }
    } catch (e) {
      console.log(`SQLite Error: ${e}`);
    }
  }
  
  await db.close();
}

checkCapabilities();
