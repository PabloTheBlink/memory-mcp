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

// --- DB helpers ---

async function renameNode(id: string, newLabel: string): Promise<void> {
  const db = await getDb();
  await db.run("UPDATE nodes SET label = ? WHERE id = ?", [newLabel, id]);
}

async function rewireEdges(fromId: string, toId: string): Promise<void> {
  const db = await getDb();
  const edges = (await db.queryAll("SELECT * FROM edges WHERE from_id = ? OR to_id = ?", [fromId, fromId]));
  for (const e of edges) {
    const src = e.from_id === fromId ? toId : e.from_id;
    const dst = e.to_id === fromId ? toId : e.to_id;
    if (src === dst) continue;
    const [a, b] = src < dst ? [src, dst] : [dst, src];
    const exists = await db.queryGet("SELECT 1 FROM edges WHERE from_id = ? AND to_id = ?", [a, b]);
    if (!exists) {
      await db.run(`
        INSERT INTO edges (from_id, to_id, weight, type, co_occurrences, created_at, last_reinforced_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `, [a, b, e.weight, e.type, e.co_occurrences, e.created_at, e.last_reinforced_at]);
    }
  }
  await db.run("DELETE FROM edges WHERE from_id = ? OR to_id = ?", [fromId, fromId]);
}

// --- Context extraction ---

async function nodeContexts(): Promise<Map<string, string[]>> {
  const edges = await getAllEdges();
  const nodes = await getAllNodes();
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

async function findDuplicates(threshold = 0.92): Promise<DuplicateCandidate[]> {
  const nodes = (await getAllNodes()).filter((n) => n.embedding && !n.label.startsWith("[ctx:"));
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
    if (a.likelyFalsePositive !== b.likelyFalsePositive)
      return a.likelyFalsePositive ? 1 : -1;
    return b.similarity - a.similarity;
  });
}

// --- HTTP helpers ---

function json(res: http.ServerResponse, data: object, status = 200) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0"
  });
  res.end(JSON.stringify(data));
}

async function readBody(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => { try { resolve(JSON.parse(data)); } catch { resolve({}); } });
  });
}

async function withJson(res: http.ServerResponse, fn: () => Promise<void>) {
  try { await fn(); json(res, { ok: true }); }
  catch (e: any) { json(res, { ok: false, error: e.message }); }
}

// --- Graph data ---

async function buildGraphData() {
  const allNodes = await getAllNodes();
  const allEdges = await getAllEdges();
  const ctxMap = await nodeContexts();

  const ctxNames = Array.from(new Set(
    allNodes.filter(n => n.label.startsWith("[ctx:")).map(n => n.label.slice(5, -1))
  ));

  const nodes = allNodes.map(n => ({
    id: n.id,
    label: n.label,
    isContext: n.label.startsWith("[ctx:"),
    isHub: n.label.startsWith("concept:"),
    contextName: n.label.startsWith("[ctx:") ? n.label.slice(5, -1) : null,
    contexts: ctxMap.get(n.id) ?? [],
    strength: n.strength,
    importance: n.importance,
    access_count: n.access_count,
    last_accessed_at: n.last_accessed_at,
    last_fired_at: n.last_fired_at,
    metadata: n.metadata,
    user_id: n.user_id,
    visibility: n.visibility,
  }));

  const nodeIds = new Set(nodes.map(n => n.id));
  const links = allEdges
    .filter(e => nodeIds.has(e.from_id) && nodeIds.has(e.to_id))
    .map(e => ({
      source: e.from_id,
      target: e.to_id,
      weight: e.weight,
      type: e.type,
      co_occurrences: e.co_occurrences,
      user_id: e.user_id,
    }));

  return { nodes, links, ctxNames };
}

// --- HTML rendering ---

function renderStyles(): string {
  return `
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #020509; overflow: hidden; font-family: 'Inter', system-ui, sans-serif; }
  #svg-container { width: 100vw; height: 100vh; }
  svg { width: 100%; height: 100%; }

  .node { cursor: pointer; }
  .node text { pointer-events: none; font-size: 10px; fill: #ccc; opacity: 0; transition: opacity 0.3s; font-weight: 300; }
  .node.hovered text, .node.selected text { opacity: 1; font-weight: 500; }
  .node.dimmed text { opacity: 0; }

  .soma { transition: r 0.3s; }
  .soma-aura { stroke-dasharray: 3,4; pointer-events: none; }
  .node.dimmed .soma { opacity: 0.06; }
  .node.dimmed .soma-aura { opacity: 0; }

  .dendrite { fill: none; stroke-linecap: round; transition: stroke-opacity 0.3s; }
  .node.dimmed .dendrite { stroke-opacity: 0.02 !important; }
  .node.hovered .dendrite, .node.selected .dendrite { stroke-opacity: 0.6 !important; }

  .link { stroke-opacity: 0.12; transition: stroke-opacity 0.3s; }
  .link.dimmed { stroke-opacity: 0.01; }
  .link.highlighted { stroke-opacity: 0.55; }

  .impulse { pointer-events: none; }

  #info-panel {
    position: fixed; top: 20px; right: 20px;
    background: rgba(10,14,24,0.92);
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 16px 20px;
    min-width: 240px;
    max-width: 320px;
    font-size: 0.82rem;
    color: #aaa;
    backdrop-filter: blur(12px);
    display: none;
    z-index: 100;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  #info-panel .label { font-size: 1rem; color: #fff; font-weight: 600; margin-bottom: 8px; word-break: break-word; }
  #info-panel .row { display: flex; justify-content: space-between; margin-bottom: 4px; }
  #info-panel .key { color: #555; }
  #info-panel .val { color: #8cf; }
  #info-panel .ctx-tag {
    display: inline-block; background: #1a2a3a; color: #6af;
    font-size: 0.72rem; padding: 2px 8px; border-radius: 10px; margin: 2px 2px 0 0;
  }
  #info-panel .visibility-tag {
    font-size: 0.65rem; padding: 1px 6px; border-radius: 4px; font-weight: bold; text-transform: uppercase;
  }
  .vis-shared { background: #059669; color: #fff; }
  .vis-private { background: #4b5563; color: #fff; }

  #nav {
    position: fixed; top: 20px; left: 20px;
    display: flex; gap: 8px; align-items: center;
  }
  #nav .title { color: #88a; font-size: 0.75rem; font-weight: 800; letter-spacing: 2px; }

  #legend {
    position: fixed; bottom: 20px; left: 20px;
    background: rgba(10,14,24,0.85); border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 12px 16px;
    font-size: 0.75rem; color: #666;
    backdrop-filter: blur(8px);
    max-height: 40vh; overflow-y: auto;
  }
  #legend h4 { color: #aaa; margin-bottom: 8px; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
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

  .user-label { position: fixed; font-size: 1.2rem; font-weight: 800; color: rgba(255,255,255,0.05); pointer-events: none; text-transform: uppercase; letter-spacing: 10px; }`;
}

function renderInfoPanel(): string {
  return `
<div id="info-panel">
  <div style="display:flex; justify-content:space-between; align-items:start">
    <div class="label" id="info-label"></div>
    <div id="info-visibility"></div>
  </div>
  <div class="row"><span class="key">Owner</span><span class="val" id="info-owner"></span></div>
  <div class="row"><span class="key">Strength</span><span class="val" id="info-strength"></span></div>
  <div class="row"><span class="key">Importance</span><span class="val" id="info-importance"></span></div>
  <div class="row"><span class="key">Accesses</span><span class="val" id="info-access"></span></div>
  <div class="row"><span class="key">Last seen</span><span class="val" id="info-last"></span></div>
  <div style="margin-top:8px; color:#555; font-size:0.72rem">Contexts / Types</div>
  <div id="info-contexts" style="margin-top:4px"></div>
</div>`;
}

function renderLegend(): string {
  return `
<div id="legend">
  <h4>Users / Entities</h4>
  <div id="legend-users"></div>
  <div class="row"><div class="swatch" style="background:#059669"></div><span>Shared / Project</span></div>

  <h4 style="margin-top:12px">Node Types</h4>
  <div class="row"><div class="swatch" style="background:#8b5cf6"></div><span>Context hub</span></div>
  <div class="row"><div class="swatch" style="background:#fbbf24"></div><span>Conceptual Hub</span></div>

  <h4 style="margin-top:12px">Link Types</h4>
  <div class="row"><div class="line-swatch" style="background:#fbbf24"></div><span>abstraction</span></div>
  <div class="row"><div class="line-swatch" style="background:#f97316"></div><span>causal</span></div>
  <div class="row"><div class="line-swatch" style="background:#38bdf8"></div><span>semantic</span></div>
  <div class="row"><div class="line-swatch" style="background:#4ade80"></div><span>temporal</span></div>
  <div class="row"><div class="line-swatch" style="background:#c084fc"></div><span>episodic</span></div>
</div>`;
}

function renderD3Script(): string {
  return `
const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select('#graph').attr('viewBox', [0,0,W,H]);
const container = svg.append('g');

// --- SVG defs: glow filters + radial gradients ---
const defs = svg.append('defs');

const GLOW_COLORS = { blue:'#38bdf8', purple:'#8b5cf6', orange:'#f97316', green:'#4ade80', white:'#ffffff', emerald:'#10b981', pink:'#ec4899', amber:'#f59e0b' };
Object.entries(GLOW_COLORS).forEach(([name, color]) => {
  // Soft outer glow
  const f = defs.append('filter').attr('id','glow-'+name).attr('x','-80%').attr('y','-80%').attr('width','260%').attr('height','260%');
  f.append('feGaussianBlur').attr('stdDeviation','5').attr('result','blur');
  const fm = f.append('feMerge');
  fm.append('feMergeNode').attr('in','blur');
  fm.append('feMergeNode').attr('in','SourceGraphic');

  // Tight impulse glow
  const fi = defs.append('filter').attr('id','iglow-'+name).attr('x','-200%').attr('y','-200%').attr('width','500%').attr('height','500%');
  fi.append('feGaussianBlur').attr('stdDeviation','2.5').attr('result','blur');
  const fim = fi.append('feMerge');
  fim.append('feMergeNode').attr('in','blur');
  fim.append('feMergeNode').attr('in','SourceGraphic');

  // Radial gradient for soma
  const grad = defs.append('radialGradient').attr('id','grad-'+name).attr('cx','35%').attr('cy','35%').attr('r','65%');
  grad.append('stop').attr('offset','0%').attr('stop-color','#ffffff').attr('stop-opacity','0.9');
  grad.append('stop').attr('offset','40%').attr('stop-color',color).attr('stop-opacity','0.85');
  grad.append('stop').attr('offset','100%').attr('stop-color',color).attr('stop-opacity','0.15');
});

const gLinks   = container.append('g').attr('class','links');
const gImpulse = container.append('g').attr('class','impulses');
const gNodes   = container.append('g').attr('class','nodes');

let nodes = [], links = [], nodeById = new Map();
let sim, link, node, selected = null;
let userColor = {}, userCenters = {};

const USER_PALETTES = ['blue','pink','amber','purple','blue'];
const PALETTE_HEX   = { blue:'#3b82f6', pink:'#ec4899', amber:'#f59e0b', purple:'#8b5cf6' };

// Pause simulation and ambient firing when tab is hidden
document.addEventListener('visibilitychange', () => {
  if (document.hidden) sim?.stop(); else sim?.restart();
});

// --- Visual helpers ---

function getNodeGradient(n) {
  if (n.visibility === 'shared') return 'url(#grad-emerald)';
  if (n.isContext)               return 'url(#grad-purple)';
  if (n.isHub)                   return 'url(#grad-amber)';
  const pal = userColor[n.user_id] || 'blue';
  return \`url(#grad-\${pal})\`;
}

function getNodeColor(n) {
  if (n.visibility === 'shared') return '#10b981';
  if (n.isContext)               return '#8b5cf6';
  if (n.isHub)                   return '#f59e0b';
  const pal = userColor[n.user_id] || 'blue';
  return PALETTE_HEX[pal] || '#94a3b8';
}

function getUserStrokeColor(n) {
  if (n.visibility === 'shared') return '#10b981';
  const pal = userColor[n.user_id] || 'blue';
  return PALETTE_HEX[pal] || '#94a3b8';
}

// Only apply expensive glow filter to special nodes — regular nodes use stroke only
function glowFilter(n) {
  if (n.visibility === 'shared') return 'url(#glow-emerald)';
  if (n.isContext)               return 'url(#glow-purple)';
  if (n.isHub)                   return 'url(#glow-amber)';
  return null;
}

function nodeRadius(n) {
  if (n.isContext) return 10 + n.access_count * 0.25;
  if (n.isHub)     return 12 + n.importance * 7;
  return 3.5 + n.strength * 10 + Math.min(n.access_count, 15) * 0.18;
}

function linkColor(type) {
  return { abstraction:'#f59e0b', causal:'#f97316', semantic:'#38bdf8', temporal:'#4ade80', episodic:'#c084fc' }[type] ?? '#334';
}

function linkGlowFilter(type) {
  return { abstraction:'iglow-amber', causal:'iglow-orange', semantic:'iglow-blue', temporal:'iglow-green', episodic:'iglow-purple' }[type] ?? 'iglow-white';
}

// --- Impulse system ---

function fireImpulse(x1, y1, x2, y2, color, duration) {
  const dot = gImpulse.append('circle')
    .attr('class','impulse')
    .attr('r', 2.5)
    .attr('cx', x1).attr('cy', y1)
    .attr('fill', '#fff')
    .attr('filter', \`url(#\${color})\`)
    .attr('opacity', 1);
  dot.transition().duration(duration).ease(d3.easeLinear)
    .attrTween('cx', () => t => x1 + (x2 - x1) * t)
    .attrTween('cy', () => t => y1 + (y2 - y1) * t)
    .attr('opacity', 0)
    .on('end', () => dot.remove());
}

function fireImpulsesFrom(d, burstCount = 1) {
  links.forEach(l => {
    const sid = l.source.id ?? l.source;
    const tid = l.target.id ?? l.target;
    if (sid !== d.id && tid !== d.id) return;
    const src = nodeById.get(sid);
    const tgt = nodeById.get(tid);
    if (!src || !tgt) return;
    const outgoing = sid === d.id;
    const [x1, y1, x2, y2] = outgoing
      ? [src.x, src.y, tgt.x, tgt.y]
      : [tgt.x, tgt.y, src.x, src.y];
    const filter = linkGlowFilter(l.type);
    const count = Math.max(1, Math.min(burstCount, 4));
    for (let i = 0; i < count; i++) {
      const delay = i * 120 + Math.random() * 60;
      const dur   = 450 + Math.random() * 250;
      setTimeout(() => fireImpulse(x1, y1, x2, y2, filter, dur), delay);
    }
  });
}

// Ambient background firing — one impulse at a time, paused when tab hidden
function startAmbientFiring() {
  // Pre-build adjacency list for O(1) lookup instead of filter() each tick
  let adjList = new Map();
  function rebuildAdj() {
    adjList = new Map();
    links.forEach(l => {
      const sid = l.source.id ?? l.source;
      if (!adjList.has(sid)) adjList.set(sid, []);
      adjList.get(sid).push(l);
    });
  }

  function tick() {
    if (!document.hidden && nodes.length > 0) {
      if (adjList.size === 0) rebuildAdj();
      // Pick a random node that has outgoing links
      const candidates = nodes.filter(n => adjList.has(n.id));
      if (candidates.length) {
        const n = candidates[Math.floor(Math.random() * candidates.length)];
        const ls = adjList.get(n.id);
        const l  = ls[Math.floor(Math.random() * ls.length)];
        const src = nodeById.get(l.source.id ?? l.source);
        const tgt = nodeById.get(l.target.id ?? l.target);
        if (src && tgt) fireImpulse(src.x, src.y, tgt.x, tgt.y, linkGlowFilter(l.type), 800 + Math.random() * 500);
      }
    }
    setTimeout(tick, 1200 + Math.random() * 2000);
  }

  // Rebuild adjacency when links change
  const _reconcileLinks = reconcileLinks;
  reconcileLinks = (newLinks, oldLinks) => {
    const result = _reconcileLinks(newLinks, oldLinks);
    adjList.clear();
    return result;
  };

  tick();
}

// --- Force simulation ---

sim = d3.forceSimulation()
  .force('link', d3.forceLink().id(d => d.id).distance(d => {
    const s = nodeById.get(d.source.id ?? d.source);
    const t = nodeById.get(d.target.id ?? d.target);
    if (s?.visibility === 'shared' || t?.visibility === 'shared') return 110;
    return (s?.user_id === t?.user_id) ? 70 : 200;
  }))
  .force('charge', d3.forceManyBody().strength(d => (d.isContext || d.isHub) ? -600 : -120))
  .force('center', d3.forceCenter(W/2, H/2))
  .force('x', d3.forceX(d => {
    if (d.visibility === 'shared' || !d.user_id) return W/2;
    return userCenters[d.user_id]?.x ?? W/2;
  }).strength(0.12))
  .force('y', d3.forceY(d => {
    if (d.visibility === 'shared' || !d.user_id) return H/2;
    return userCenters[d.user_id]?.y ?? H/2;
  }).strength(0.05))
  .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 18));

svg.call(d3.zoom().scaleExtent([0.05, 5]).on('zoom', e => container.attr('transform', e.transform)));

// --- Highlight / selection ---

function highlight(d) {
  const connectedIds = new Set([d.id]);
  links.forEach(l => {
    const sid = l.source.id ?? l.source;
    const tid = l.target.id ?? l.target;
    if (sid === d.id) connectedIds.add(tid);
    if (tid === d.id) connectedIds.add(sid);
  });
  node.classed('dimmed',      n => !connectedIds.has(n.id));
  node.classed('hovered',     n => n.id === d.id);
  link.classed('dimmed',      l => (l.source.id??l.source)!==d.id && (l.target.id??l.target)!==d.id);
  link.classed('highlighted', l => (l.source.id??l.source)===d.id || (l.target.id??l.target)===d.id);
  showInfoPanel(d);
  fireImpulsesFrom(d, 2);
}

function clearHighlight() {
  node.classed('dimmed hovered selected', false);
  link.classed('dimmed highlighted', false);
  document.getElementById('info-panel').style.display = 'none';
  selected = null;
}

function showInfoPanel(d) {
  document.getElementById('info-panel').style.display = 'block';
  document.getElementById('info-label').textContent = d.isContext ? d.contextName : (d.isHub ? d.label.replace('concept:','') : d.label);
  document.getElementById('info-owner').textContent  = d.user_id ? d.user_id.slice(0,8)+'...' : 'System';
  document.getElementById('info-strength').textContent   = d.strength.toFixed(3);
  document.getElementById('info-importance').textContent = (d.importance || 0.5).toFixed(3);
  document.getElementById('info-access').textContent     = d.access_count;
  document.getElementById('info-last').textContent       = new Date(d.last_accessed_at).toLocaleDateString();
  document.getElementById('info-contexts').innerHTML     = d.contexts.map(c => '<span class="ctx-tag">'+c+'</span>').join('') || '<span style="color:#444">none</span>';
  document.getElementById('info-visibility').innerHTML   = \`<span class="visibility-tag vis-\${d.visibility}">\${d.visibility}</span>\`;
}

// --- User registration ---

function assignUserCenters(userIds) {
  userIds.forEach((uid, i) => {
    if (userColor[uid]) return;
    userColor[uid] = USER_PALETTES[i % USER_PALETTES.length];
    const angle = (i / userIds.length) * Math.PI * 2;
    userCenters[uid] = { x: W/2 + Math.cos(angle) * W * 0.3, y: H/2 + Math.sin(angle) * H * 0.15 };
    if (userIds.length === 2) {
      userCenters[userIds[0]] = { x: W * 0.22, y: H/2 };
      userCenters[userIds[1]] = { x: W * 0.78, y: H/2 };
      document.getElementById('user-a-label').textContent = 'BRAIN ' + userIds[0].slice(-4);
      document.getElementById('user-b-label').textContent = 'BRAIN ' + userIds[1].slice(-4);
    } else {
      document.getElementById('user-a-label').style.display = 'none';
      document.getElementById('user-b-label').style.display = 'none';
    }
    const hex = PALETTE_HEX[userColor[uid]] || '#94a3b8';
    d3.select('#legend-users').append('div').attr('class','row').html(
      \`<div class="swatch" style="background:\${hex}"></div><span>User \${uid.slice(0,6)}</span>\`
    );
  });
}

// --- Node/link reconciliation ---

function reconcileNodes(newNodes, oldNodeById) {
  nodeById = new Map();
  return newNodes.map(nn => {
    const existing = oldNodeById.get(nn.id);
    if (existing) {
      if (nn.last_fired_at > (existing.last_fired_at || 0)) existing._shouldSpike = true;
      Object.assign(existing, nn);
      nodeById.set(nn.id, existing);
      return existing;
    }
    nn.x = W/2 + (Math.random()-0.5)*220;
    nn.y = H/2 + (Math.random()-0.5)*220;
    nn._isNew = true;
    nodeById.set(nn.id, nn);
    return nn;
  });
}

function reconcileLinks(newLinks, oldLinks) {
  // O(n) via Map instead of O(n²) via find()
  const oldMap = new Map(oldLinks.map(l => [(l.source.id||l.source)+'-'+(l.target.id||l.target), l]));
  return newLinks.map(nl => {
    const existing = oldMap.get(\`\${nl.source}-\${nl.target}\`);
    return existing ? Object.assign(existing, nl) : nl;
  });
}

// --- Dendrite generation (called once on enter) ---

function addDendrites(g) {
  g.each(function(d) {
    const sel = d3.select(this).append('g').attr('class','dendrites');
    const r = nodeRadius(d);
    const count = d.isContext ? 9 : d.isHub ? 7 : 4 + Math.round(d.strength * 3);
    const color = getUserStrokeColor(d);
    for (let i = 0; i < count; i++) {
      const angle  = (i / count) * Math.PI * 2 + (Math.random() - 0.5) * 0.5;
      const mainLen = r * (1.6 + Math.random() * 1.4);
      const x2 = Math.cos(angle) * mainLen;
      const y2 = Math.sin(angle) * mainLen;
      // Main process (axon or dendrite)
      sel.append('line').attr('class','dendrite')
        .attr('x1', Math.cos(angle) * r * 0.9).attr('y1', Math.sin(angle) * r * 0.9)
        .attr('x2', x2).attr('y2', y2)
        .attr('stroke', color).attr('stroke-width', 0.7).attr('stroke-opacity', 0.28);
      // Terminal branches
      const branches = 1 + Math.floor(Math.random() * 2);
      for (let b = 0; b < branches; b++) {
        const ba  = angle + (Math.random() - 0.5) * 1.4;
        const bx  = x2 + Math.cos(ba) * mainLen * (0.25 + Math.random() * 0.25);
        const by  = y2 + Math.sin(ba) * mainLen * (0.25 + Math.random() * 0.25);
        sel.append('line').attr('class','dendrite')
          .attr('x1', x2).attr('y1', y2)
          .attr('x2', bx).attr('y2', by)
          .attr('stroke', color).attr('stroke-width', 0.35).attr('stroke-opacity', 0.18);
      }
    }
  });
}

// --- DOM rendering ---

function renderLinks() {
  link = gLinks.selectAll('.link').data(links, d => \`\${d.source.id||d.source}-\${d.target.id||d.target}\`)
    .join('line').attr('class','link')
    .attr('stroke', d => linkColor(d.type))
    .attr('stroke-width', d => Math.max(0.4, d.weight * 1.8));
}

function renderNodes() {
  node = gNodes.selectAll('.node').data(nodes, d => d.id)
    .join(
      enter => {
        const g = enter.append('g').attr('class','node').attr('id', d => \`node-\${d.id}\`);
        addDendrites(g);
        g.append('circle').attr('class','soma-aura');
        g.append('circle').attr('class','soma');
        g.append('text');
        return g;
      },
      update => update,
      exit => exit.transition().duration(500).style('opacity',0).remove()
    )
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) sim.alphaTarget(0.2).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

  // Soma (cell body) — solid gradient sphere
  node.select('.soma')
    .attr('r',      d => nodeRadius(d) * 0.55)
    .attr('fill',   d => getNodeGradient(d))
    .attr('filter', d => glowFilter(d));

  // Soma aura — thin dashed ring
  node.select('.soma-aura')
    .attr('r',            d => nodeRadius(d))
    .attr('fill',         'none')
    .attr('stroke',       d => getUserStrokeColor(d))
    .attr('stroke-width', 0.6)
    .attr('stroke-opacity', 0.2);

  node.select('text')
    .attr('dy', d => -(nodeRadius(d) + 8))
    .attr('text-anchor','middle')
    .text(d => d.isContext ? d.contextName : d.label);

  node.on('mouseenter', (e,d) => { if (!selected) highlight(d); })
      .on('mouseleave', ()    => { if (!selected) clearHighlight(); })
      .on('click',      (e,d) => {
        e.stopPropagation();
        if (selected === d.id) { clearHighlight(); return; }
        selected = d.id;
        node.classed('selected', n => n.id === d.id);
        highlight(d);
      });
}

function animateNewNodes() {
  nodes.forEach(n => {
    if (!n._isNew && !n._shouldSpike) return;
    const soma = gNodes.select(\`#node-\${n.id}\`).select('.soma');
    if (soma.empty()) return;
    const baseR  = nodeRadius(n) * 0.55;
    const isNew  = n._isNew;
    soma.interrupt()
      .attr('r', baseR * 4)
      .attr('fill', '#ffffff')
      .attr('filter', 'url(#glow-white)')
      .transition().duration(isNew ? 900 : 500).ease(d3.easeQuadOut)
      .attr('r',      baseR)
      .attr('fill',   getNodeGradient(n))
      .attr('filter', glowFilter(n));
    fireImpulsesFrom(n, isNew ? 3 : 2);
    n._isNew = false;
    n._shouldSpike = false;
  });
}

// --- Main update loop ---

async function update() {
  const data = await fetch('/api/graph').then(r => r.json());
  const { nodes: newNodes, links: newLinks } = data;

  const userIds = Array.from(new Set(newNodes.filter(n => n.user_id).map(n => n.user_id)));
  assignUserCenters(userIds);

  const oldNodeById = nodeById;
  nodes = reconcileNodes(newNodes, oldNodeById);
  links = reconcileLinks(newLinks, links);

  renderLinks();
  renderNodes();
  animateNewNodes();

  sim.nodes(nodes);
  sim.force('link').links(links);
  sim.alpha(0.05).restart();
}

// --- Tick & search ---

sim.on('tick', () => {
  if (!link || !node) return;
  link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
      .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  node.attr('transform', d => \`translate(\${d.x},\${d.y})\`);
});

svg.on('click', clearHighlight);
document.getElementById('search-input').addEventListener('input', e => {
  const q = e.target.value.toLowerCase().trim();
  if (!q) { node.classed('dimmed',false); link.classed('dimmed',false); return; }
  node.classed('dimmed', n => !n.label.toLowerCase().includes(q));
  link.classed('dimmed', true);
});

update();
setInterval(update, 2000);
startAmbientFiring();`;
}

function renderBrain(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Memory Brain Cluster</title>
<style>${renderStyles()}</style>
</head>
<body>
<div id="svg-container"><svg id="graph"></svg></div>
<div id="nav"><span class="title">NEURAL NETWORK • MULTI-USER</span></div>
${renderInfoPanel()}
${renderLegend()}
<div id="user-a-label" class="user-label" style="top:50%; left:20%; transform:translate(-50%,-50%) rotate(-90deg)">BRAIN A</div>
<div id="user-b-label" class="user-label" style="top:50%; left:80%; transform:translate(-50%,-50%) rotate(90deg)">BRAIN B</div>
<div id="shared-label" class="user-label" style="top:10%; left:50%; transform:translate(-50%,-50%)">SHARED NUCLEUS</div>
<div id="search"><input id="search-input" placeholder="Search neurons…" /></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>${renderD3Script()}</script>
</body>
</html>`;
}

// --- Route handlers ---

async function handleGetGraph(res: http.ServerResponse) {
  json(res, await buildGraphData());
}

async function handleDeleteNode(res: http.ServerResponse, body: any) {
  await withJson(res, () => deleteNode(body.id));
}

async function handleDeleteEdge(res: http.ServerResponse, body: any) {
  await withJson(res, () => deleteEdge(body.from_id, body.to_id));
}

async function handleRename(res: http.ServerResponse, body: any) {
  await withJson(res, () => renameNode(body.id, body.label));
}

async function handleMerge(res: http.ServerResponse, body: any) {
  await withJson(res, async () => {
    await rewireEdges(body.delete_id, body.keep_id);
    await deleteNode(body.delete_id);
  });
}

// --- Server ---

const POST_ROUTES: Record<string, (res: http.ServerResponse, body: any) => Promise<void>> = {
  "/delete-node": handleDeleteNode,
  "/delete-edge": handleDeleteEdge,
  "/rename": handleRename,
  "/merge": handleMerge,
};

const server = http.createServer(async (req, res) => {
  const url = req.url ?? "/";

  if (req.method === "GET" && url === "/") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(renderBrain());
    return;
  }

  if (req.method === "GET" && url === "/api/graph") {
    await handleGetGraph(res);
    return;
  }

  if (req.method === "POST") {
    const handler = POST_ROUTES[url];
    if (handler) {
      const body = await readBody(req);
      await handler(res, body);
      return;
    }
  }

  res.writeHead(404);
  res.end();
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Memory Review → http://localhost:${PORT}`);
});
