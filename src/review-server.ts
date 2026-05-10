import http from "http";
import {
  getAllNodes,
  getAllEdges,
  getNodeById,
  deleteNode,
  deleteEdge,
  getDb,
  MemoryNode,
  MemoryEdge,
} from "./graph";
import { cosineSimilarity } from "./embeddings";

const PORT = 3131;

// --- DB helpers not in graph.ts ---

function renameNode(id: string, newLabel: string): void {
  getDb().prepare("UPDATE nodes SET label = ? WHERE id = ?").run(newLabel, id);
}

function rewireEdges(fromId: string, toId: string): void {
  const db = getDb();
  const edges = (db.prepare("SELECT * FROM edges WHERE from_id = ? OR to_id = ?").all(fromId, fromId) as any[]);
  for (const e of edges) {
    const src = e.from_id === fromId ? toId : e.from_id;
    const dst = e.to_id === fromId ? toId : e.to_id;
    if (src === dst) continue;
    const [a, b] = src < dst ? [src, dst] : [dst, src];
    const exists = db.prepare("SELECT 1 FROM edges WHERE from_id = ? AND to_id = ?").get(a, b);
    if (!exists) {
      db.prepare(`
        INSERT INTO edges (from_id, to_id, weight, type, co_occurrences, created_at, last_reinforced_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `).run(a, b, e.weight, e.type, e.co_occurrences, e.created_at, e.last_reinforced_at);
    }
  }
  db.prepare("DELETE FROM edges WHERE from_id = ? OR to_id = ?").run(fromId, fromId);
}

// --- Context extraction from episodic edges ---

function nodeContexts(): Map<string, string[]> {
  const edges = getAllEdges();
  const nodes = getAllNodes();
  const ctxLabels = new Map(
    nodes.filter((n) => n.label.startsWith("[ctx:")).map((n) => [n.id, n.label.slice(5, -1)])
  );
  const map = new Map<string, string[]>();
  for (const e of edges) {
    if (e.type === "episodic") {
      const ctxId = ctxLabels.has(e.to_id) ? e.to_id : ctxLabels.has(e.from_id) ? e.from_id : null;
      const memId = ctxLabels.has(e.to_id) ? e.from_id : ctxLabels.has(e.from_id) ? e.to_id : null;
      if (ctxId && memId) {
        const ctx = ctxLabels.get(ctxId)!;
        const existing = map.get(memId) ?? [];
        if (!existing.includes(ctx)) map.set(memId, [...existing, ctx]);
      }
    }
  }
  return map;
}

// --- Duplicate detection ---

interface DuplicateCandidate {
  a: MemoryNode;
  b: MemoryNode;
  similarity: number;
  likelyFalsePositive: boolean;
  reason: string;
}

// Levenshtein distance to catch "pablo" vs "Pablo" type duplicates
// and avoid flagging completely unrelated short labels
function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
  return dp[m][n];
}

function findDuplicates(threshold = 0.92): DuplicateCandidate[] {
  const nodes = getAllNodes().filter((n) => n.embedding && !n.label.startsWith("[ctx:"));
  const results: DuplicateCandidate[] = [];

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const sim = cosineSimilarity(nodes[i].embedding!, nodes[j].embedding!);
      if (sim < threshold) continue;

      const la = nodes[i].label.toLowerCase();
      const lb = nodes[j].label.toLowerCase();
      const lev = levenshtein(la, lb);
      const maxLen = Math.max(la.length, lb.length);
      const textSimilarity = 1 - lev / maxLen;

      // If labels look nothing alike textually, the embedding similarity is a model artifact
      // (short/proper nouns often get degenerate vectors in nomic-embed-text)
      const likelyFalsePositive = textSimilarity < 0.35;

      let reason = "";
      if (likelyFalsePositive) {
        reason = `Embedding artifact — labels share no text similarity (${Math.round(textSimilarity * 100)}% text match)`;
      } else if (textSimilarity > 0.8) {
        reason = `Very similar text (${Math.round(textSimilarity * 100)}% text match)`;
      } else {
        reason = `Same semantic meaning, different wording`;
      }

      results.push({ a: nodes[i], b: nodes[j], similarity: sim, likelyFalsePositive, reason });
    }
  }

  return results.sort((a, b) => {
    // Real duplicates first, false positives last
    if (a.likelyFalsePositive !== b.likelyFalsePositive)
      return a.likelyFalsePositive ? 1 : -1;
    return b.similarity - a.similarity;
  });
}

// --- HTML ---

function renderPage(): string {
  const nodes = getAllNodes().filter((n) => !n.label.startsWith("[ctx:"));
  const edges = getAllEdges();
  const contexts = nodeContexts();
  const duplicates = findDuplicates(0.92);

  const grouped = new Map<string, MemoryNode[]>();
  for (const n of nodes) {
    const ctxList = contexts.get(n.id) ?? ["(none)"];
    for (const ctx of ctxList) {
      const arr = grouped.get(ctx) ?? [];
      arr.push(n);
      grouped.set(ctx, arr);
    }
  }

  const ctxSections = Array.from(grouped.entries())
    .sort(([a], [b]) => a === "(none)" ? 1 : b === "(none)" ? -1 : a.localeCompare(b))
    .map(([ctx, ns]) => `
      <details ${ctx !== "(none)" ? "open" : ""} class="${ctx === "(none)" ? "ctx-none" : ""}">
        <summary><b>${ctx === "(none)" ? "⚪ (no context — pre-existing nodes)" : esc(ctx)}</b> <span class="badge">${ns.length}</span></summary>
        <table>
          <thead><tr><th>Label</th><th>Strength</th><th>Accesses</th><th>Last seen</th><th>Actions</th></tr></thead>
          <tbody>
            ${ns.sort((a, b) => b.strength - a.strength).map((n) => `
            <tr id="row-${n.id}">
              <td>
                <span class="label-text" id="lbl-${n.id}">${esc(n.label)}</span>
                <input class="label-edit hidden" id="inp-${n.id}" value="${esc(n.label)}">
              </td>
              <td>${n.strength.toFixed(3)}</td>
              <td>${n.access_count}</td>
              <td>${new Date(n.last_accessed_at).toLocaleDateString()}</td>
              <td class="actions">
                <button onclick="startEdit('${n.id}')">✏️</button>
                <button class="save-btn hidden" onclick="saveEdit('${n.id}')">💾</button>
                <button onclick="deleteNode('${n.id}')">🗑️</button>
              </td>
            </tr>`).join("")}
          </tbody>
        </table>
      </details>`).join("");

  const realDups = duplicates.filter((d) => !d.likelyFalsePositive);
  const maybeDups = duplicates.filter((d) => d.likelyFalsePositive);

  const dupRows = (dups: typeof duplicates) => dups.slice(0, 30).map((d) => `
    <tr class="${d.likelyFalsePositive ? 'fp-row' : ''}">
      <td>${esc(d.a.label)}</td>
      <td>${esc(d.b.label)}</td>
      <td>${(d.similarity * 100).toFixed(1)}%</td>
      <td class="reason">${esc(d.reason)}</td>
      <td class="actions">
        ${d.likelyFalsePositive
          ? `<span class="fp-label">Probably not duplicates</span>`
          : `<button onclick="mergeNodes('${d.a.id}','${d.b.id}')">Keep "${esc(d.a.label)}"</button>
             <button onclick="mergeNodes('${d.b.id}','${d.a.id}')">Keep "${esc(d.b.label)}"</button>`
        }
      </td>
    </tr>`).join("");

  const edgeRows = edges
    .filter((e) => {
      const from = getNodeById(e.from_id);
      const to = getNodeById(e.to_id);
      return from && to && !from.label.startsWith("[ctx:") && !to.label.startsWith("[ctx:");
    })
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 200)
    .map((e) => {
      const from = getNodeById(e.from_id)!;
      const to = getNodeById(e.to_id)!;
      return `<tr>
        <td>${esc(from.label)}</td>
        <td><span class="tag">${e.type}</span></td>
        <td>${esc(to.label)}</td>
        <td>${e.weight.toFixed(3)}</td>
        <td>${e.co_occurrences}</td>
        <td><button onclick="deleteEdge('${e.from_id}','${e.to_id}')">🗑️</button></td>
      </tr>`;
    }).join("");

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Memory Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 24px; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; color: #fff; }
  .subtitle { color: #666; font-size: 0.85rem; margin-bottom: 24px; }
  h2 { font-size: 1rem; color: #aaa; margin: 32px 0 12px; text-transform: uppercase; letter-spacing: 0.08em; }
  details { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; margin-bottom: 12px; }
  summary { padding: 12px 16px; cursor: pointer; font-size: 0.9rem; user-select: none; }
  summary:hover { background: #222; }
  .badge { background: #333; color: #aaa; font-size: 0.75rem; padding: 2px 7px; border-radius: 10px; margin-left: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 8px 12px; color: #666; font-weight: 500; border-bottom: 1px solid #222; }
  td { padding: 7px 12px; border-bottom: 1px solid #1e1e1e; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1e1e1e; }
  button { background: #2a2a2a; color: #ccc; border: 1px solid #333; padding: 3px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; margin-right: 4px; }
  button:hover { background: #333; color: #fff; }
  .actions button { padding: 2px 8px; }
  input.label-edit { background: #111; color: #fff; border: 1px solid #444; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; width: 100%; }
  .hidden { display: none; }
  .tag { background: #1e3a2a; color: #4caf82; font-size: 0.75rem; padding: 1px 6px; border-radius: 4px; }
  #toast { position: fixed; bottom: 24px; right: 24px; background: #1a3a1a; color: #4caf82; border: 1px solid #2a5a2a; padding: 10px 16px; border-radius: 8px; font-size: 0.85rem; display: none; }
  .stats { display: flex; gap: 16px; margin-bottom: 24px; }
  .stat { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 12px 20px; }
  .stat-n { font-size: 1.6rem; font-weight: 600; color: #fff; }
  .stat-l { font-size: 0.75rem; color: #666; margin-top: 2px; }
  #dup-section { display: ${duplicates.length > 0 ? "block" : "none"}; }
  .fp-row td { opacity: 0.45; }
  .fp-label { font-size: 0.78rem; color: #666; font-style: italic; }
  .reason { font-size: 0.78rem; color: #888; max-width: 260px; }
  .ctx-none summary { color: #666; }
  .ctx-none summary b { font-weight: 400; }
  .help { font-size: 0.8rem; color: #555; margin: -8px 0 20px; line-height: 1.6; }
  .help b { color: #888; }
</style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px">
  <h1>Memory Review</h1>
  <a href="/brain" style="background:#1a1a2e;border:1px solid #2a2a4a;color:#8b5cf6;font-size:0.82rem;padding:6px 14px;border-radius:8px;text-decoration:none">🧠 Brain view</a>
</div>
<p class="subtitle">Manual review interface — changes take effect immediately in the graph</p>

<div class="stats">
  <div class="stat"><div class="stat-n">${nodes.length}</div><div class="stat-l">Memory nodes</div></div>
  <div class="stat"><div class="stat-n">${edges.length}</div><div class="stat-l">Associations</div></div>
  <div class="stat"><div class="stat-n">${grouped.size}</div><div class="stat-l">Contexts</div></div>
  <div class="stat"><div class="stat-n" style="color:${realDups.length > 0 ? '#e88' : '#4caf82'}">${realDups.length}</div><div class="stat-l">Real duplicates</div></div>
</div>

<div id="dup-section">
  <h2>⚠️ Duplicate candidates</h2>
  ${realDups.length > 0 ? `
  <details open>
    <summary><b>Likely duplicates</b> <span class="badge">${realDups.length}</span></summary>
    <table>
      <thead><tr><th>Node A</th><th>Node B</th><th>Similarity</th><th>Why</th><th>Actions</th></tr></thead>
      <tbody>${dupRows(realDups)}</tbody>
    </table>
  </details>` : ""}
  ${maybeDups.length > 0 ? `
  <details>
    <summary style="color:#555"><b>Probably not duplicates</b> — similar embedding, different meaning <span class="badge">${maybeDups.length}</span></summary>
    <table>
      <thead><tr><th>Node A</th><th>Node B</th><th>Similarity</th><th>Why</th><th></th></tr></thead>
      <tbody>${dupRows(maybeDups)}</tbody>
    </table>
  </details>` : ""}
</div>

<h2>Nodes by context</h2>
<p class="help">
  <b>Context</b> = where a memory was formed. Memories in <b>(none)</b> existed before context tracking was added — they work fine but aren't biased toward any project.
</p>
${ctxSections || "<p style='color:#666;padding:16px'>No nodes yet.</p>"}

<h2>Associations (top 200 by weight)</h2>
<details>
  <summary>Edges <span class="badge">${edges.length}</span></summary>
  <table>
    <thead><tr><th>From</th><th>Type</th><th>To</th><th>Weight</th><th>Co-occ.</th><th></th></tr></thead>
    <tbody>${edgeRows}</tbody>
  </table>
</details>

<div id="toast"></div>

<script>
function toast(msg, ok = true) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = ok ? '#1a3a1a' : '#3a1a1a';
  t.style.color = ok ? '#4caf82' : '#e88';
  t.style.borderColor = ok ? '#2a5a2a' : '#5a2a2a';
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2500);
}

async function api(path, body) {
  const r = await fetch(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  return r.json();
}

function startEdit(id) {
  document.getElementById('lbl-' + id).classList.add('hidden');
  document.getElementById('inp-' + id).classList.remove('hidden');
  document.querySelector('#row-' + id + ' .save-btn').classList.remove('hidden');
}

async function saveEdit(id) {
  const val = document.getElementById('inp-' + id).value.trim();
  if (!val) return;
  const r = await api('/rename', { id, label: val });
  if (r.ok) { toast('Renamed'); setTimeout(() => location.reload(), 600); }
  else toast(r.error, false);
}

async function deleteNode(id) {
  if (!confirm('Delete this node and all its associations?')) return;
  const r = await api('/delete-node', { id });
  if (r.ok) { document.getElementById('row-' + id)?.remove(); toast('Deleted'); }
  else toast(r.error, false);
}

async function deleteEdge(from_id, to_id) {
  const r = await api('/delete-edge', { from_id, to_id });
  if (r.ok) toast('Edge deleted');
  else toast(r.error, false);
  setTimeout(() => location.reload(), 600);
}

async function mergeNodes(deleteId, keepId) {
  const r = await api('/merge', { delete_id: deleteId, keep_id: keepId });
  if (r.ok) { toast('Merged'); setTimeout(() => location.reload(), 800); }
  else toast(r.error, false);
}
</script>
</body>
</html>`;
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function json(res: http.ServerResponse, data: object, status = 200) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

async function readBody(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => { try { resolve(JSON.parse(data)); } catch { resolve({}); } });
  });
}

function buildGraphData() {
  const allNodes = getAllNodes();
  const allEdges = getAllEdges();
  const ctxMap = nodeContexts();

  // Assign a context color index to each unique context
  const ctxNames = Array.from(new Set(
    allNodes.filter(n => n.label.startsWith("[ctx:")).map(n => n.label.slice(5, -1))
  ));

  const nodes = allNodes.map(n => ({
    id: n.id,
    label: n.label,
    isContext: n.label.startsWith("[ctx:"),
    contextName: n.label.startsWith("[ctx:") ? n.label.slice(5, -1) : null,
    contexts: ctxMap.get(n.id) ?? [],
    strength: n.strength,
    access_count: n.access_count,
    last_accessed_at: n.last_accessed_at,
  }));

  const links = allEdges.map(e => ({
    source: e.from_id,
    target: e.to_id,
    weight: e.weight,
    type: e.type,
    co_occurrences: e.co_occurrences,
  }));

  return { nodes, links, ctxNames };
}

function renderBrain(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Memory Brain</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #060a12; overflow: hidden; font-family: system-ui, sans-serif; }
  #svg-container { width: 100vw; height: 100vh; }
  svg { width: 100%; height: 100%; }

  .node circle { cursor: pointer; transition: r 0.2s; }
  .node text { pointer-events: none; font-size: 11px; fill: #ccc; opacity: 0; transition: opacity 0.2s; }
  .node.hovered text, .node.selected text { opacity: 1; }
  .node.dimmed circle { opacity: 0.08; }
  .node.dimmed text { opacity: 0; }

  .link { stroke-opacity: 0.35; transition: stroke-opacity 0.2s; }
  .link.dimmed { stroke-opacity: 0.03; }
  .link.highlighted { stroke-opacity: 0.9; }

  #info-panel {
    position: fixed; top: 20px; right: 20px;
    background: rgba(10,14,24,0.92);
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 16px 20px;
    min-width: 220px;
    max-width: 300px;
    font-size: 0.82rem;
    color: #aaa;
    backdrop-filter: blur(8px);
    display: none;
  }
  #info-panel .label { font-size: 1rem; color: #fff; font-weight: 600; margin-bottom: 8px; word-break: break-word; }
  #info-panel .row { display: flex; justify-content: space-between; margin-bottom: 4px; }
  #info-panel .key { color: #555; }
  #info-panel .val { color: #8cf; }
  #info-panel .ctx-tag {
    display: inline-block; background: #1a2a3a; color: #6af;
    font-size: 0.72rem; padding: 2px 8px; border-radius: 10px; margin: 2px 2px 0 0;
  }

  #nav {
    position: fixed; top: 20px; left: 20px;
    display: flex; gap: 8px; align-items: center;
  }
  #nav a {
    background: rgba(10,14,24,0.85); border: 1px solid #1e2a3a;
    color: #888; font-size: 0.8rem; padding: 6px 14px; border-radius: 6px;
    text-decoration: none; backdrop-filter: blur(8px);
  }
  #nav a:hover { color: #fff; border-color: #334; }
  #nav .title { color: #446; font-size: 0.75rem; }

  #legend {
    position: fixed; bottom: 20px; left: 20px;
    background: rgba(10,14,24,0.85); border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 12px 16px;
    font-size: 0.75rem; color: #555;
    backdrop-filter: blur(8px);
  }
  #legend .row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
  #legend .swatch { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  #legend .line-swatch { width: 24px; height: 2px; flex-shrink: 0; }

  #search {
    position: fixed; bottom: 20px; right: 20px;
    background: rgba(10,14,24,0.85); border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 8px 12px;
    backdrop-filter: blur(8px);
  }
  #search input {
    background: transparent; border: none; outline: none;
    color: #ccc; font-size: 0.85rem; width: 180px;
  }
  #search input::placeholder { color: #333; }
</style>
</head>
<body>
<div id="svg-container"><svg id="graph"></svg></div>

<div id="nav">
  <a href="/">← Review</a>
  <span class="title">MEMORY BRAIN</span>
</div>

<div id="info-panel">
  <div class="label" id="info-label"></div>
  <div class="row"><span class="key">Strength</span><span class="val" id="info-strength"></span></div>
  <div class="row"><span class="key">Accesses</span><span class="val" id="info-access"></span></div>
  <div class="row"><span class="key">Last seen</span><span class="val" id="info-last"></span></div>
  <div style="margin-top:8px; color:#555; font-size:0.72rem">Contexts</div>
  <div id="info-contexts" style="margin-top:4px"></div>
</div>

<div id="legend">
  <div class="row"><div class="swatch" style="background:#8b5cf6"></div><span>Context hub</span></div>
  <div class="row"><div class="swatch" style="background:#38bdf8"></div><span>user context</span></div>
  <div id="legend-projects"></div>
  <div class="row"><div class="swatch" style="background:#94a3b8"></div><span>no context</span></div>
  <div style="margin-top:8px; border-top:1px solid #1e2a3a; padding-top:8px">
    <div class="row"><div class="line-swatch" style="background:#f97316"></div><span>causal</span></div>
    <div class="row"><div class="line-swatch" style="background:#38bdf8"></div><span>semantic</span></div>
    <div class="row"><div class="line-swatch" style="background:#4ade80"></div><span>temporal</span></div>
    <div class="row"><div class="line-swatch" style="background:#c084fc"></div><span>episodic</span></div>
  </div>
</div>

<div id="search"><input id="search-input" placeholder="Search neurons…" /></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select('#graph').attr('viewBox', [0,0,W,H]);

// Glow filter
const defs = svg.append('defs');
['blue','purple','orange','green','white'].forEach((name, i) => {
  const colors = { blue:'#38bdf8', purple:'#8b5cf6', orange:'#f97316', green:'#4ade80', white:'#e2e8f0' };
  const f = defs.append('filter').attr('id', 'glow-'+name).attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%');
  f.append('feGaussianBlur').attr('stdDeviation','4').attr('result','blur');
  const merge = f.append('feMerge');
  merge.append('feMergeNode').attr('in','blur');
  merge.append('feMergeNode').attr('in','SourceGraphic');
});

// Project colors palette
const PROJECT_COLORS = ['#34d399','#fb923c','#f472b6','#facc15','#a78bfa','#22d3ee'];

fetch('/api/graph').then(r => r.json()).then(({ nodes, links, ctxNames }) => {
  // Map context name → color
  const ctxColor = {};
  ctxNames.forEach((name, i) => {
    ctxColor[name] = name === 'user' ? '#38bdf8' : PROJECT_COLORS[i % PROJECT_COLORS.length];
  });

  // Legend projects
  ctxNames.filter(n => n !== 'user').forEach(name => {
    d3.select('#legend-projects').append('div').attr('class','row').html(
      \`<div class="swatch" style="background:\${ctxColor[name]}"></div><span>\${name}</span>\`
    );
  });

  function nodeColor(n) {
    if (n.isContext) return '#8b5cf6';
    if (n.contexts.length > 0) {
      const c = n.contexts.find(c => c !== 'user') ?? n.contexts[0];
      return ctxColor[c] ?? '#94a3b8';
    }
    return '#94a3b8';
  }

  function nodeRadius(n) {
    if (n.isContext) return 14 + n.access_count * 0.4;
    return 5 + n.strength * 14 + Math.min(n.access_count, 20) * 0.3;
  }

  function linkColor(type) {
    return { causal:'#f97316', semantic:'#38bdf8', temporal:'#4ade80', episodic:'#c084fc' }[type] ?? '#888';
  }

  // Build id sets for fast lookup
  const nodeById = new Map(nodes.map(n => [n.id, n]));

  // Simulation
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(d => {
      const s = nodeById.get(d.source.id ?? d.source);
      const t = nodeById.get(d.target.id ?? d.target);
      if (s?.isContext || t?.isContext) return 80;
      return 120 / (d.weight + 0.1);
    }).strength(d => d.weight * 0.4))
    .force('charge', d3.forceManyBody().strength(d => d.isContext ? -400 : -120))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 8));

  const container = svg.append('g');

  // Zoom + pan
  svg.call(d3.zoom().scaleExtent([0.2, 4]).on('zoom', e => container.attr('transform', e.transform)));

  // Links
  const link = container.append('g').selectAll('line')
    .data(links).join('line')
    .attr('class','link')
    .attr('stroke', d => linkColor(d.type))
    .attr('stroke-width', d => Math.max(0.5, d.weight * 3));

  // Nodes
  const node = container.append('g').selectAll('g')
    .data(nodes).join('g')
    .attr('class','node')
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; })
    );

  node.append('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => nodeColor(d) + (d.isContext ? '22' : '18'))
    .attr('stroke', d => nodeColor(d))
    .attr('stroke-width', d => d.isContext ? 2 : 1.5)
    .attr('filter', d => d.isContext ? 'url(#glow-purple)' : 'url(#glow-blue)');

  // Pulse ring for context nodes
  node.filter(d => d.isContext).append('circle')
    .attr('r', d => nodeRadius(d) + 6)
    .attr('fill', 'none')
    .attr('stroke', d => nodeColor(d))
    .attr('stroke-width', 0.5)
    .attr('opacity', 0.3)
    .attr('stroke-dasharray', '3 3');

  node.append('text')
    .attr('dy', d => -(nodeRadius(d) + 6))
    .attr('text-anchor','middle')
    .text(d => d.isContext ? d.contextName : d.label);

  // Hover / click
  let selected = null;

  function highlight(d) {
    const connectedIds = new Set([d.id]);
    links.forEach(l => {
      const sid = l.source.id ?? l.source;
      const tid = l.target.id ?? l.target;
      if (sid === d.id) connectedIds.add(tid);
      if (tid === d.id) connectedIds.add(sid);
    });

    node.classed('dimmed', n => !connectedIds.has(n.id));
    node.classed('hovered', n => n.id === d.id);
    link.classed('dimmed',      l => (l.source.id??l.source)!==d.id && (l.target.id??l.target)!==d.id);
    link.classed('highlighted', l => (l.source.id??l.source)===d.id || (l.target.id??l.target)===d.id);

    // Info panel
    const panel = document.getElementById('info-panel');
    panel.style.display = 'block';
    document.getElementById('info-label').textContent = d.isContext ? d.contextName : d.label;
    document.getElementById('info-strength').textContent = d.strength.toFixed(3);
    document.getElementById('info-access').textContent = d.access_count;
    document.getElementById('info-last').textContent = new Date(d.last_accessed_at).toLocaleDateString();
    document.getElementById('info-contexts').innerHTML =
      d.isContext
        ? \`<span class="ctx-tag">context hub</span>\`
        : (d.contexts.length ? d.contexts.map(c => \`<span class="ctx-tag">\${c}</span>\`).join('') : '<span style="color:#444">none</span>');
  }

  function clearHighlight() {
    node.classed('dimmed hovered selected', false);
    link.classed('dimmed highlighted', false);
    document.getElementById('info-panel').style.display = 'none';
    selected = null;
  }

  node.on('mouseenter', (e,d) => { if (!selected) highlight(d); })
      .on('mouseleave', () => { if (!selected) clearHighlight(); })
      .on('click', (e,d) => {
        e.stopPropagation();
        if (selected === d.id) { clearHighlight(); return; }
        selected = d.id;
        node.classed('selected', n => n.id === d.id);
        highlight(d);
      });

  svg.on('click', clearHighlight);

  // Search
  document.getElementById('search-input').addEventListener('input', e => {
    const q = e.target.value.toLowerCase().trim();
    if (!q) { node.classed('dimmed', false); link.classed('dimmed', false); return; }
    node.classed('dimmed', n => !n.label.toLowerCase().includes(q));
    link.classed('dimmed', true);
  });

  // Tick
  sim.on('tick', () => {
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
        .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    node.attr('transform', d => \`translate(\${d.x},\${d.y})\`);
  });

  // Animate pulse on context nodes
  function pulse() {
    container.selectAll('circle[stroke-dasharray]')
      .transition().duration(2000).ease(d3.easeSinInOut)
      .attr('r', function() { return parseFloat(d3.select(this.parentNode).datum().isContext ? nodeRadius(d3.select(this.parentNode).datum()) + 12 : 0); })
      .attr('opacity', 0.05)
      .transition().duration(2000).ease(d3.easeSinInOut)
      .attr('r', function() { return parseFloat(d3.select(this.parentNode).datum().isContext ? nodeRadius(d3.select(this.parentNode).datum()) + 6 : 0); })
      .attr('opacity', 0.3)
      .on('end', pulse);
  }
  pulse();
});
</script>
</body>
</html>`;
}

const server = http.createServer(async (req, res) => {
  const url = req.url ?? "/";

  if (req.method === "GET" && url === "/") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(renderPage());
    return;
  }

  if (req.method === "GET" && url === "/brain") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(renderBrain());
    return;
  }

  if (req.method === "GET" && url === "/api/graph") {
    json(res, buildGraphData());
    return;
  }

  if (req.method === "POST") {
    const body = await readBody(req);

    if (url === "/delete-node") {
      try { deleteNode(body.id); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/delete-edge") {
      try { deleteEdge(body.from_id, body.to_id); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/rename") {
      try { renameNode(body.id, body.label); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/merge") {
      try {
        rewireEdges(body.delete_id, body.keep_id);
        deleteNode(body.delete_id);
        json(res, { ok: true });
      } catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }
  }

  res.writeHead(404);
  res.end();
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Memory Review → http://localhost:${PORT}`);
});
