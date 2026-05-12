import http from "http";
import { WebSocketServer, WebSocket } from "ws";
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
  body { background: #00060f; overflow: hidden; font-family: 'Outfit', 'Inter', system-ui, sans-serif; color: #e6edf3; }
  #svg-container { position: relative; width: 100vw; height: 100vh; }
  svg { position: relative; width: 100%; height: 100%; z-index: 1; }
  #impulse-canvas { position: absolute; top: 0; left: 0; pointer-events: none; z-index: 2; }

  .node { cursor: pointer; }
  .node text {
    pointer-events: none; font-size: 11px; fill: #7aa2c8; opacity: 0;
    transition: opacity 0.3s; font-weight: 600; letter-spacing: 0.3px;
    paint-order: stroke; stroke: #00060f; stroke-width: 4px; stroke-linejoin: round;
  }
  .node.hovered text, .node.selected text { opacity: 1; fill: #e2f0ff; }
  .node.dimmed text { opacity: 0 !important; }

  .soma { transition: opacity 0.4s; }
  .node.dimmed .soma { opacity: 0.04; }

  .aura { pointer-events: none; animation: aura-breathe 3.5s ease-in-out infinite; }
  .node.hovered .aura, .node.selected .aura { animation: aura-active 1s ease-in-out infinite; }
  .node.dimmed .aura { opacity: 0 !important; }
  @keyframes aura-breathe { 0%,100% { opacity: 0.08; } 50% { opacity: 0.22; } }
  @keyframes aura-active  { 0%,100% { opacity: 0.4;  } 50% { opacity: 0.7;  } }

  .link { fill: none; transition: stroke-opacity 0.3s, stroke-width 0.3s; }
  .link.dimmed { stroke-opacity: 0.01 !important; }
  .link.highlighted { stroke-opacity: 0.9 !important; stroke-width: 2.5px !important; }

  #info-panel {
    position: fixed; top: 24px; right: 24px;
    background: rgba(8, 12, 20, 0.88);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 22px;
    min-width: 290px; max-width: 350px;
    backdrop-filter: blur(24px);
    display: none; z-index: 100;
    box-shadow: 0 0 0 1px rgba(99,179,237,0.08), 0 20px 60px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.06);
  }
  #info-panel .label { font-size: 1rem; color: #f0f6fc; font-weight: 700; margin-bottom: 14px; line-height: 1.4; word-break: break-word; }
  #info-panel .row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.72rem; }
  #info-panel .key { color: #4a5568; text-transform: uppercase; letter-spacing: 0.8px; }
  #info-panel .val { color: #63b3ed; font-weight: 600; }
  #info-panel .ctx-tag {
    display: inline-block; background: rgba(99,179,237,0.1); color: #63b3ed;
    font-size: 0.68rem; padding: 3px 9px; border-radius: 20px; margin: 3px 3px 0 0;
    border: 1px solid rgba(99,179,237,0.18);
  }
  #info-panel .divider { border: none; border-top: 1px solid rgba(255,255,255,0.05); margin: 12px 0; }

  #nav { position: fixed; top: 24px; left: 24px; display: flex; flex-direction: column; gap: 5px; }
  #nav .title { color: #fff; font-size: 0.75rem; font-weight: 900; letter-spacing: 5px; text-transform: uppercase; opacity: 0.6; }
  #nav .status { color: #3fb950; font-size: 0.62rem; font-weight: 700; text-transform: uppercase; display: flex; align-items: center; gap: 7px; letter-spacing: 1px; }
  #nav .status::before { content: ''; width: 5px; height: 5px; background: #3fb950; border-radius: 50%; box-shadow: 0 0 10px #3fb950, 0 0 20px #3fb950; animation: pulse-dot 2s infinite; }
  @keyframes pulse-dot { 0%,100% { opacity: 1; box-shadow: 0 0 10px #3fb950, 0 0 20px #3fb950; } 50% { opacity: 0.5; box-shadow: 0 0 4px #3fb950; } }

  #legend {
    position: fixed; bottom: 24px; left: 24px;
    background: rgba(8,12,20,0.82); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 14px 16px;
    font-size: 0.72rem; color: #6b7280; backdrop-filter: blur(16px);
    max-height: 40vh; overflow-y: auto;
  }
  #legend h4 { color: #374151; margin-bottom: 8px; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px; }
  #legend .row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
  .swatch { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .line-swatch { width: 18px; height: 2px; border-radius: 2px; flex-shrink: 0; }
  `;
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
const defs = svg.append('defs');

// ── SVG filters ─────────────────────────────────────────────────────────────
const LINK_COLORS = { abstraction:'#f59e0b', causal:'#f97316', semantic:'#38bdf8', temporal:'#4ade80', episodic:'#c084fc' };
const LINK_FILTERS = { abstraction:'iglow-amber', causal:'iglow-orange', semantic:'iglow-blue', temporal:'iglow-green', episodic:'iglow-purple' };

// Node glow — double blur for rich halo
const ng = defs.append('filter').attr('id','node-glow').attr('x','-120%').attr('y','-120%').attr('width','340%').attr('height','340%');
ng.append('feGaussianBlur').attr('in','SourceGraphic').attr('stdDeviation','7').attr('result','blur1');
ng.append('feGaussianBlur').attr('in','SourceGraphic').attr('stdDeviation','3').attr('result','blur2');
const ngm = ng.append('feMerge');
ngm.append('feMergeNode').attr('in','blur1');
ngm.append('feMergeNode').attr('in','blur2');
ngm.append('feMergeNode').attr('in','SourceGraphic');

// Impulse glow filters for links
Object.entries({ amber:'#f59e0b', orange:'#f97316', blue:'#38bdf8', green:'#4ade80', purple:'#c084fc', white:'#fff' }).forEach(([name]) => {
  const fi = defs.append('filter').attr('id','iglow-'+name).attr('x','-150%').attr('y','-150%').attr('width','400%').attr('height','400%');
  fi.append('feGaussianBlur').attr('stdDeviation','2').attr('result','b');
  const m = fi.append('feMerge');
  m.append('feMergeNode').attr('in','b');
  m.append('feMergeNode').attr('in','SourceGraphic');
});

const NODE_COLORS = { blue:'#3b82f6', pink:'#ec4899', amber:'#f59e0b', purple:'#8b5cf6', emerald:'#10b981', teal:'#06b6d4' };

const gLinks = container.append('g').attr('class','links');
const gNodes = container.append('g').attr('class','nodes');


// ── Impulse canvas ──────────────────────────────────────────────────────────
const impulseCanvas = document.createElement('canvas');
impulseCanvas.id = 'impulse-canvas'; impulseCanvas.width = W; impulseCanvas.height = H;
document.getElementById('svg-container').appendChild(impulseCanvas);
const ictx = impulseCanvas.getContext('2d');

const IMPULSE_COLORS = { 'iglow-amber':'#f59e0b','iglow-orange':'#f97316','iglow-blue':'#38bdf8','iglow-green':'#4ade80','iglow-purple':'#c084fc','iglow-white':'#fff' };
const MAX_IMPULSES = 20;
let activeImpulses = [];
let rafId = null;

// No shadowBlur — fake glow with layered strokes (much faster)
function strokeGlow(x1,y1,x2,y2,color,baseWidth,alpha,k) {
  ictx.lineCap = 'round';
  ictx.strokeStyle = color;
  ictx.globalAlpha = alpha * 0.18; ictx.lineWidth = baseWidth*9*k;
  ictx.beginPath(); ictx.moveTo(x1,y1); ictx.lineTo(x2,y2); ictx.stroke();
  ictx.globalAlpha = alpha * 0.35; ictx.lineWidth = baseWidth*4*k;
  ictx.beginPath(); ictx.moveTo(x1,y1); ictx.lineTo(x2,y2); ictx.stroke();
  ictx.strokeStyle = '#fff';
  ictx.globalAlpha = alpha * 0.9;  ictx.lineWidth = baseWidth*k;
  ictx.beginPath(); ictx.moveTo(x1,y1); ictx.lineTo(x2,y2); ictx.stroke();
}

function rafLoop() {
  ictx.clearRect(0,0,W,H);
  const tr = d3.zoomTransform(svg.node());
  activeImpulses = activeImpulses.filter(i => i.elapsed < i.duration);

  for (const imp of activeImpulses) {
    if (imp.type === 'bloom') {
      const p = d3.easeCubicOut(imp.elapsed / imp.duration);
      const [bx,by] = tr.apply([imp.x, imp.y]);
      ictx.save();
      // Outer ring
      ictx.globalAlpha = (1-p) * 0.4;
      ictx.strokeStyle = imp.color; ictx.lineWidth = 5*(1-p)*tr.k;
      ictx.beginPath(); ictx.arc(bx,by,(4+p*35)*tr.k,0,Math.PI*2); ictx.stroke();
      // Inner ring
      ictx.globalAlpha = (1-p) * 0.7;
      ictx.lineWidth = 2*(1-p)*tr.k;
      ictx.beginPath(); ictx.arc(bx,by,(2+p*16)*tr.k,0,Math.PI*2); ictx.stroke();
      ictx.restore();
      imp.elapsed += 16; continue;
    }

    const p = d3.easeQuadIn(imp.elapsed / imp.duration);
    const cx = imp.x1+(imp.x2-imp.x1)*p, cy = imp.y1+(imp.y2-imp.y1)*p;
    const dx = imp.x2-imp.x1, dy = imp.y2-imp.y1;
    const len = Math.sqrt(dx*dx+dy*dy)||1;
    const nx = -dy/len, ny = dx/len;
    const wobble = Math.sin(imp.elapsed*0.25+imp.phase)*4*(1-p)*Math.min(p*5,1);
    const wcx = cx+nx*wobble, wcy = cy+ny*wobble;
    const tailP = Math.max(0,p-0.3);
    const [sx,sy]   = tr.apply([wcx,wcy]);
    const [stx,sty] = tr.apply([imp.x1+(imp.x2-imp.x1)*tailP,imp.y1+(imp.y2-imp.y1)*tailP]);
    const alpha = p < 0.88 ? 1 : (1-p)/0.12;
    const k = tr.k;

    ictx.save();
    strokeGlow(stx,sty,sx,sy,imp.color,1.2,alpha,k);

    // Core orb — layered circles, no shadowBlur
    const orb = ictx.createRadialGradient(sx,sy,0,sx,sy,7*k);
    orb.addColorStop(0,'#fff'); orb.addColorStop(0.4,imp.color); orb.addColorStop(1,'transparent');
    ictx.globalAlpha = alpha; ictx.fillStyle = orb;
    ictx.beginPath(); ictx.arc(sx,sy,7*k,0,Math.PI*2); ictx.fill();

    ictx.restore();

    if (p > 0.87 && !imp.bloomed) {
      imp.bloomed = true;
      activeImpulses.push({type:'bloom',x:imp.x2,y:imp.y2,color:imp.color,elapsed:0,duration:320});
    }
    imp.elapsed += 16;
  }
  rafId = activeImpulses.length > 0 ? requestAnimationFrame(rafLoop) : null;
}

// ── State ───────────────────────────────────────────────────────────────────
let nodes = [], links = [], nodeById = new Map();
let sim, link, node, selected = null;
let userColor = {};
const USER_PALETTES = ['blue','pink','teal','purple','amber'];

function getNodeColor(n) {
  if (n.visibility === 'shared') return NODE_COLORS.emerald;
  if (n.isContext) return NODE_COLORS.purple;
  if (n.isHub) return NODE_COLORS.amber;
  return NODE_COLORS[userColor[n.user_id] || 'blue'] || NODE_COLORS.blue;
}

function nodeRadius(n) {
  const d = n._degree || 0;
  if (n.isContext || n.isHub) return 12 + Math.sqrt(d) * 10;
  return 5 + Math.sqrt(d) * 4 + (n.strength||0) * 2;
}

function fireImpulse(x1,y1,x2,y2,filterName,duration) {
  if (activeImpulses.length >= MAX_IMPULSES) return;
  activeImpulses.push({ x1,y1,x2,y2, color:IMPULSE_COLORS[filterName]??'#fff', duration, elapsed:0, phase:Math.random()*Math.PI*2, bloomed:false });
  if (!rafId) rafId = requestAnimationFrame(rafLoop);
}

function fireImpulsesFrom(d, burstCount=1) {
  links.forEach(l => {
    const sid = l.source.id??l.source, tid = l.target.id??l.target;
    if (sid!==d.id && tid!==d.id) return;
    const src = nodeById.get(sid), tgt = nodeById.get(tid);
    if (!src||!tgt) return;
    const out = sid===d.id;
    const [x1,y1,x2,y2] = out ? [src.x,src.y,tgt.x,tgt.y] : [tgt.x,tgt.y,src.x,src.y];
    const filter = LINK_FILTERS[l.type]??'iglow-white';
    for (let i=0; i<burstCount; i++) setTimeout(()=>fireImpulse(x1,y1,x2,y2,filter,480+Math.random()*220), i*110);
  });
}

// ── Zoom ────────────────────────────────────────────────────────────────────
const zoom = d3.zoom().scaleExtent([0.04,5]).on('zoom', e => container.attr('transform', e.transform));
svg.call(zoom);

function zoomToFit(duration=1000) {
  const nucleus = nodes.reduce((a,b) => (b._degree||0) > (a._degree||0) ? b : a, nodes[0]);
  if (!nucleus || nucleus.x == null) return;
  const scale = 0.7;
  const tx = W/2 - scale * nucleus.x;
  const ty = H/2 - scale * nucleus.y;
  svg.transition().duration(duration).ease(d3.easeCubicInOut)
    .call(zoom.transform, d3.zoomIdentity.translate(tx,ty).scale(scale));
}

// ── Simulation ──────────────────────────────────────────────────────────────
sim = d3.forceSimulation()
  .force('link',    d3.forceLink().id(d=>d.id).distance(d => 80 + nodeRadius(d.source) + nodeRadius(d.target)).strength(0.3))
  .force('charge',  d3.forceManyBody().strength(d => -(500 + nodeRadius(d)*12)))
  .force('center',  d3.forceCenter(W/2, H/2))
  .force('collide', d3.forceCollide().radius(d=>nodeRadius(d)+35).strength(0.9));

let fitDone = false;

// ── Highlight ───────────────────────────────────────────────────────────────
function highlight(d) {
  const ids = new Set([d.id]);
  links.forEach(l => { if(l.source.id===d.id) ids.add(l.target.id); if(l.target.id===d.id) ids.add(l.source.id); });
  node.classed('dimmed', n=>!ids.has(n.id)).classed('selected', n=>n.id===d.id);
  link.classed('dimmed', l=>l.source.id!==d.id&&l.target.id!==d.id);
  link.classed('highlighted', l=>l.source.id===d.id||l.target.id===d.id);
  const p = document.getElementById('info-panel');
  p.style.display = 'block';
  document.getElementById('info-label').textContent = d.label;
  document.getElementById('info-owner').textContent = d.user_id || '—';
  document.getElementById('info-strength').textContent = (d.strength??0).toFixed(3);
  document.getElementById('info-importance').textContent = (d.importance??0).toFixed(3);
  document.getElementById('info-access').textContent = d.access_count??0;
  document.getElementById('info-last').textContent = d.last_accessed_at ? new Date(d.last_accessed_at*1000).toLocaleDateString() : '—';
  document.getElementById('info-contexts').innerHTML = (d.contexts||[]).map(c=>\`<span class="ctx-tag">\${c}</span>\`).join('');
  fireImpulsesFrom(d, 3);
}

// ── WebSocket ───────────────────────────────────────────────────────────────
function connectWS() {
  const proto = window.location.protocol==='https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(\`\${proto}//\${window.location.host}\`);
  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type==='graph') updateGraph(msg.data);
    if (msg.type==='highlight') msg.ids.forEach(id => { const d=nodeById.get(id); if(d) fireImpulsesFrom(d,3); });
  };
  ws.onerror = () => ws.close();
  ws.onclose = () => setTimeout(connectWS, 2000);
}

// ── Update graph ─────────────────────────────────────────────────────────────
function updateGraph(data) {
  const users = [...new Set(data.nodes.map(n=>n.user_id).filter(Boolean))];
  users.forEach((u,i) => { if (!userColor[u]) userColor[u] = USER_PALETTES[i%USER_PALETTES.length]; });

  const oldNodes = new Map(nodes.map(n=>[n.id,n]));
  nodes = data.nodes.map(n => Object.assign(oldNodes.get(n.id)||{x:W/2,y:H/2}, n));
  links = data.links;
  nodeById = new Map(nodes.map(n=>[n.id,n]));

  // Degree centrality — drives node size
  nodes.forEach(n => n._degree = 0);
  links.forEach(l => {
    const sid = l.source.id??l.source, tid = l.target.id??l.target;
    const s = nodeById.get(sid), t = nodeById.get(tid);
    if (s) s._degree = (s._degree||0) + 1;
    if (t) t._degree = (t._degree||0) + 1;
  });

  link = gLinks.selectAll('.link').data(links)
    .join('line')
    .attr('class', d=>'link link-'+d.type)
    .attr('stroke', d=>LINK_COLORS[d.type]??'#1e3a5f')
    .attr('stroke-width', d=>0.8+(d.weight||0.5)*1.2)
    .attr('stroke-opacity', 0.55)
    .attr('filter', d=>{ const f=LINK_FILTERS[d.type]; return f?'url(#'+f+')':null; });

  node = gNodes.selectAll('.node').data(nodes, d=>d.id)
    .join(enter => {
      const g = enter.append('g').attr('class','node').style('opacity',0);
      g.append('circle').attr('class','aura');
      g.append('circle').attr('class','soma');
      g.append('text').attr('text-anchor','middle');
      g.transition().duration(700).delay((_,i)=>i*6).style('opacity',1);
      return g;
    })
    .call(d3.drag()
      .on('start',(e,d)=>{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag', (e,d)=>{ d.fx=e.x; d.fy=e.y; })
      .on('end',  (e,d)=>{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }))
    .on('click',      (event,d)=>{ event.stopPropagation(); selected=d; highlight(d); })
    .on('mouseenter', (e,d)=>d3.select(e.currentTarget).classed('hovered',true))
    .on('mouseleave', (e,d)=>d3.select(e.currentTarget).classed('hovered',false));

  const r = d=>nodeRadius(d), c = d=>getNodeColor(d);
  node.select('.aura').attr('r', d=>r(d)*2.4).attr('fill','none').attr('stroke',c).attr('stroke-width',1.5).attr('stroke-opacity',0.2);
  node.select('.soma').attr('r', r).attr('fill', c).attr('filter','url(#node-glow)');
  node.select('text').attr('dy', d=>r(d)+15).text(d=>d.label);

  sim.nodes(nodes);
  sim.force('link').links(links);
  sim.alpha(0.5).restart();
  if (!fitDone) {
    sim.on('end.fit', ()=>{ fitDone=true; zoomToFit(); sim.on('end.fit',null); });
    setTimeout(()=>{ if(!fitDone){ fitDone=true; zoomToFit(); } }, 3000);
  }
}

sim.on('tick', () => {
  if (link) link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  if (node) node.attr('transform', d=>\`translate(\${d.x},\${d.y})\`);
});

svg.on('click', ()=>{
  if (node) node.classed('dimmed',false).classed('selected',false);
  if (link) { link.classed('dimmed',false); link.classed('highlighted',false); }
  document.getElementById('info-panel').style.display = 'none';
  selected = null;
});

connectWS();
`;
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

<div id="nav">
  <span class="title">CORTICAL CORE</span>
  <span class="status">Neural Socket Active</span>
</div>
${renderInfoPanel()}
${renderLegend()}
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

const POST_ROUTES: Record<string, (res: http.ServerResponse, body: any) => Promise<void>> = {
  "/delete-node": handleDeleteNode,
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

const wss = new WebSocketServer({ server });

function broadcast(data: any) {
  const message = JSON.stringify({ type: "graph", data });
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) client.send(message);
  });
}

function broadcastHighlight(ids: string[]) {
  const message = JSON.stringify({ type: "highlight", ids });
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) client.send(message);
  });
}

wss.on("connection", async (ws) => {
  const data = await buildGraphData();
  ws.send(JSON.stringify({ type: "graph", data }));
});

let lastHash = "";
let lastFiredMap = new Map<string, number | null>();

setInterval(async () => {
  try {
    const data = await buildGraphData();
    const currentHash = JSON.stringify(data);

    const newlyFired: string[] = [];
    for (const n of data.nodes) {
      const prev = lastFiredMap.get(n.id);
      if (n.last_fired_at && n.last_fired_at !== prev) newlyFired.push(n.id);
      lastFiredMap.set(n.id, n.last_fired_at ?? null);
    }

    if (currentHash !== lastHash) {
      lastHash = currentHash;
      broadcast(data);
    }
    if (newlyFired.length > 0) broadcastHighlight(newlyFired);
  } catch (e) {}
}, 2000);
