import { getDb } from "../src/graph";
const db = getDb();
const result = db.pragma("foreign_keys");
console.log("Foreign keys status:", result);
