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
  rewireEdges,
} from "./graph";
import { cosineSimilarity } from "./embeddings";

const PORT = 3131;

// --- DB helpers not in graph.ts ---

async function renameNode(id: string, newLabel: string): Promise<void> {
  const db = await getDb();
  await db.run("UPDATE nodes SET label = ? WHERE id = ?", [newLabel, id]);
}


// --- Context extraction from episodic edges ---

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

// --- HTML ---

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

function renderBrain(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Memory Brain Cluster</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #060a12; overflow: hidden; font-family: 'Inter', system-ui, sans-serif; }
  #svg-container { width: 100vw; height: 100vh; }
  svg { width: 100%; height: 100%; }

  .node circle { cursor: pointer; transition: r 0.2s; }
  .node text { pointer-events: none; font-size: 10px; fill: #ccc; opacity: 0; transition: opacity 0.2s; font-weight: 300; }
  .node.hovered text, .node.selected text { opacity: 1; font-weight: 500; }
  .node.dimmed circle { opacity: 0.08; }
  .node.dimmed text { opacity: 0; }

  .link { 
    stroke-opacity: 0.35; 
    stroke-dasharray: 2, 8;
    animation: neural-flow 30s linear infinite;
    transition: stroke-opacity 0.2s, stroke-width 0.2s;
  }
  @keyframes neural-flow {
    from { stroke-dashoffset: 100; }
    to { stroke-dashoffset: 0; }
  }
  .link.dimmed { stroke-opacity: 0.05; }
  .link.highlighted { 
    stroke-opacity: 0.95; 
    stroke-width: 3px; 
    stroke-dasharray: none; 
    animation: none;
  }
  .node.firing circle, .node.firing path {
    stroke-opacity: 1;
    stroke-width: 3px;
    stroke: #fff !important;
    filter: url(#glow-white);
  }

  /* Differential effect for contexts */
  .node.context path {
    stroke: #8b5cf6;
    stroke-width: 2px;
    animation: context-pulse 3s ease-in-out infinite;
  }
  @keyframes context-pulse {
    0% { stroke-width: 2px; stroke-opacity: 0.4; }
    50% { stroke-width: 5px; stroke-opacity: 0.9; }
    100% { stroke-width: 2px; stroke-opacity: 0.4; }
  }
  .node.context text { opacity: 0.5; font-weight: 600; fill: #8b5cf6; }

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

  .user-label { position: fixed; font-size: 1.2rem; font-weight: 800; color: rgba(255,255,255,0.05); pointer-events: none; text-transform: uppercase; letter-spacing: 10px; }
</style>
</head>
<body>
<div id="svg-container"><svg id="graph"></svg></div>

<div id="nav"><span class="title">NEURAL NETWORK • MULTI-USER</span></div>

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
</div>

<div id="legend">
  <h4>Users / Entities</h4>
  <div id="legend-users"></div>
  <div class="row"><div class="swatch" style="background:#059669"></div><span>Shared / Project</span></div>
  
  <h4 style="margin-top:12px">Node Types</h4>
  <div class="row"><div class="swatch" style="background:#8b5cf6"></div><span>Context hub</span></div>
  <div class="row"><div class="swatch" style="background:#fbbf24"></div><span>Conceptual Hub</span></div>
  <div class="row"><div class="swatch" style="background:#ef4444"></div><span>Conflict</span></div>
  <div class="row"><div class="swatch" style="background:#ec4899"></div><span>Curiosity Gap</span></div>
  
  <h4 style="margin-top:12px">Link Types</h4>
  <div class="row"><div class="line-swatch" style="background:#fbbf24"></div><span>abstraction</span></div>
  <div class="row"><div class="line-swatch" style="background:#f97316"></div><span>causal</span></div>
  <div class="row"><div class="line-swatch" style="background:#38bdf8"></div><span>semantic</span></div>
  <div class="row"><div class="line-swatch" style="background:#4ade80"></div><span>temporal</span></div>
  <div class="row"><div class="line-swatch" style="background:#c084fc"></div><span>episodic</span></div>
</div>

<div id="user-a-label" class="user-label" style="top:50%; left:20%; transform:translate(-50%,-50%) rotate(-90deg)">BRAIN A</div>
<div id="user-b-label" class="user-label" style="top:50%; left:80%; transform:translate(-50%,-50%) rotate(90deg)">BRAIN B</div>
<div id="shared-label" class="user-label" style="top:10%; left:50%; transform:translate(-50%,-50%)">SHARED NUCLEUS</div>

<div id="search"><input id="search-input" placeholder="Search neurons…" /></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const hexagon = (r) => {
  const points = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 3) * i - Math.PI / 6;
    points.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return 'M' + points.map(p => p.join(',')).join('L') + 'Z';
};

const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select('#graph').attr('viewBox', [0,0,W,H]);
const container = svg.append('g');

const defs = svg.append('defs');
['blue','purple','orange','green','white','emerald'].forEach(name => {
  const colors = { blue:'#38bdf8', purple:'#8b5cf6', orange:'#f97316', green:'#4ade80', white:'#ffffff', emerald:'#10b981' };
  const f = defs.append('filter').attr('id', 'glow-'+name).attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%');
  f.append('feGaussianBlur').attr('stdDeviation','4').attr('result','blur');
  const merge = f.append('feMerge');
  merge.append('feMergeNode').attr('in','blur');
  merge.append('feMergeNode').attr('in','SourceGraphic');
});

const gLinks = container.append('g').attr('class', 'links');
const gNodes = container.append('g').attr('class', 'nodes');

let nodes = [], links = [], nodeById = new Map();
let sim, link, node, selected = null;
let userColor = {};
let userCenters = {};
const USER_PALETTES = [
  ['#3b82f6', '#60a5fa', '#93c5fd'], // Blue
  ['#ec4899', '#f472b6', '#fbcfe8'], // Pink
  ['#f59e0b', '#fbbf24', '#fcd34d'], // Amber
  ['#8b5cf6', '#a78bfa', '#c4b5fd'], // Violet
  ['#06b6d4', '#22d3ee', '#67e8f9'], // Cyan
];

function getNodeColor(n) {
  if (n.label.startsWith('conflict:')) return '#ef4444'; // Red for conflicts
  if (n.label.startsWith('curiosity:')) return '#ec4899'; // Pink for curiosity
  if (n.visibility === 'shared') return '#10b981'; // Emerald for shared
  if (n.isContext) return '#8b5cf6';             // Purple for context
  if (n.isHub) return '#fbbf24';                 // Amber for hubs
  if (n.user_id) {
    return userColor[n.user_id] || '#94a3b8';
  }
  return '#94a3b8';
}

function getUserStroke(n) {
  if (n.visibility === 'shared') return '#10b981';
  return userColor[n.user_id] || '#94a3b8';
}

function nodeRadius(n) {
  if (n.isContext) return 12 + n.access_count * 0.3;
  if (n.isHub) return 14 + n.importance * 8;
  return 4 + n.strength * 12 + Math.min(n.access_count, 15) * 0.2;
}

function linkColor(type) {
  return { abstraction:'#fbbf24', causal:'#f97316', semantic:'#38bdf8', temporal:'#4ade80', episodic:'#c084fc' }[type] ?? '#444';
}

sim = d3.forceSimulation()
  .force('link', d3.forceLink().id(d => d.id).distance(d => {
    const s = nodeById.get(d.source.id ?? d.source);
    const t = nodeById.get(d.target.id ?? d.target);
    if (s?.visibility === 'shared' || t?.visibility === 'shared') return 100;
    return (s?.user_id === t?.user_id) ? 60 : 180;
  }))
  .force('charge', d3.forceManyBody().strength(d => (d.isContext || d.isHub) ? -500 : -100))
  .force('center', d3.forceCenter(W/2, H/2))
  .force('x', d3.forceX(d => {
    if (d.visibility === 'shared') return W/2;
    if (!d.user_id) return W/2;
    return userCenters[d.user_id]?.x ?? W/2;
  }).strength(0.15))
  .force('y', d3.forceY(d => {
    if (d.visibility === 'shared') return H/2;
    if (!d.user_id) return H/2;
    return userCenters[d.user_id]?.y ?? H/2;
  }).strength(0.05))
  .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 10));

const zoom = d3.zoom().scaleExtent([0.05, 5]).on('zoom', e => container.attr('transform', e.transform));
svg.call(zoom);
svg.call(zoom.transform, d3.zoomIdentity.translate(W/2, H/2).scale(0.6).translate(-W/2, -H/2));

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
  link.classed('dimmed', l => (l.source.id??l.source)!==d.id && (l.target.id??l.target)!==d.id);
  link.classed('highlighted', l => (l.source.id??l.source)===d.id || (l.target.id??l.target)===d.id);

  const panel = document.getElementById('info-panel');
  panel.style.display = 'block';
  document.getElementById('info-label').textContent = d.isContext ? d.contextName : (d.isHub ? d.label.replace('concept:','') : d.label);
  document.getElementById('info-owner').textContent = d.user_id ? d.user_id.slice(0,8) + '...' : 'System';
  document.getElementById('info-strength').textContent = d.strength.toFixed(3);
  document.getElementById('info-importance').textContent = (d.importance || 0.5).toFixed(3);
  document.getElementById('info-access').textContent = d.access_count;
  document.getElementById('info-last').textContent = new Date(d.last_accessed_at).toLocaleDateString();
  document.getElementById('info-contexts').innerHTML = d.contexts.map(c => '<span class="ctx-tag">' + c + '</span>').join('') || '<span style="color:#444">none</span>';
  
  const vis = document.getElementById('info-visibility');
  vis.innerHTML = \`<span class="visibility-tag vis-\${d.visibility}">\${d.visibility}</span>\`;
}

function clearHighlight() {
  node.classed('dimmed hovered selected', false);
  link.classed('dimmed highlighted', false);
  document.getElementById('info-panel').style.display = 'none';
  selected = null;
}

async function update() {
  const data = await fetch('/api/graph').then(r => r.json());
  const { nodes: newNodes, links: newLinks } = data;

  // Identify users and assign centers/colors
  const userIds = Array.from(new Set(newNodes.filter(n => n.user_id).map(n => n.user_id)));
  userIds.forEach((uid, i) => {
    if (!userColor[uid]) {
      userColor[uid] = USER_PALETTES[i % USER_PALETTES.length][0];
      // Position users in a circle or line
      const angle = (i / userIds.length) * Math.PI * 2;
      const radius = W * 0.3;
      userCenters[uid] = {
        x: W/2 + Math.cos(angle) * radius,
        y: H/2 + Math.sin(angle) * radius * 0.5
      };
      // For exactly two brains, use left and right
      if (userIds.length === 2) {
        userCenters[userIds[0]] = { x: W * 0.2, y: H/2 };
        userCenters[userIds[1]] = { x: W * 0.8, y: H/2 };
        document.getElementById('user-a-label').textContent = 'BRAIN ' + userIds[0].slice(-4);
        document.getElementById('user-b-label').textContent = 'BRAIN ' + userIds[1].slice(-4);
      } else {
        document.getElementById('user-a-label').style.display = 'none';
        document.getElementById('user-b-label').style.display = 'none';
      }

      d3.select('#legend-users').append('div').attr('class','row').html(
        \`<div class="swatch" style="background:\${userColor[uid]}"></div><span>User \${uid.slice(0,6)}</span>\`
      );
    }
  });

  const oldNodeById = nodeById;
  nodeById = new Map();
  nodes = newNodes.map(nn => {
    const existing = oldNodeById.get(nn.id);
    if (existing) {
      if (nn.last_fired_at > (existing.last_fired_at || 0)) existing._shouldSpike = true;
      Object.assign(existing, nn);
      nodeById.set(nn.id, existing);
      return existing;
    } else {
      nn.x = W/2 + (Math.random()-0.5)*200;
      nn.y = H/2 + (Math.random()-0.5)*200;
      nn._isNew = true;
      nodeById.set(nn.id, nn);
      return nn;
    }
  });

  links = newLinks.map(nl => {
    const existing = links.find(l => (l.source.id || l.source) === nl.source && (l.target.id || l.target) === nl.target);
    return existing ? Object.assign(existing, nl) : nl;
  });

  link = gLinks.selectAll('.link').data(links, d => \`\${d.source.id || d.source}-\${d.target.id || d.target}\`)
    .join('line').attr('class','link')
    .attr('stroke', d => linkColor(d.type))
    .attr('stroke-width', d => d.type === 'abstraction' ? 4 : Math.max(1.5, d.weight * 4))
    .attr('stroke-dasharray', d => d.type === 'abstraction' ? 'none' : '2, 8');

  node = gNodes.selectAll('.node').data(nodes, d => d.id)
    .join(
      enter => {
        const g = enter.append('g').attr('class', d => 'node' + (d.isContext ? ' context' : ''));
        g.attr('id', d => \`node-\${d.id}\`);
        g.append('path').filter(d => d.isContext);
        g.append('circle').filter(d => !d.isContext);
        g.append('text');
        return g;
      },
      update => {
        update.attr('class', d => 'node' + (d.isContext ? ' context' : ''));
        return update;
      },
      exit => exit.transition().duration(500).style('opacity', 0).remove()
    )
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) sim.alphaTarget(0.2).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

  node.filter(':not(.context)').select('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => getNodeColor(d) + '44')
    .attr('stroke', 'none')
    .attr('stroke-width', 0)
    .attr('filter', d => {
        if (d.visibility === 'shared') return 'url(#glow-emerald)';
        const uid = d.user_id;
        const paletteIndex = Array.from(new Set(nodes.map(n => n.user_id))).indexOf(uid);
        const glowNames = ['blue','purple','orange','green','white','emerald'];
        return \`url(#glow-\${glowNames[paletteIndex % glowNames.length]})\`;
    });

  node.filter('.context').select('path')
    .attr('d', d => hexagon(nodeRadius(d)))
    .attr('fill', d => getNodeColor(d) + '66')
    .attr('filter', 'url(#glow-purple)');

  node.select('text')
    .attr('dy', d => -(nodeRadius(d) + 5))
    .attr('text-anchor','middle')
    .text(d => d.isContext ? d.contextName : d.label);

  node.on('mouseenter', (e,d) => { if (!selected) highlight(d); })
      .on('mouseleave', () => { if (!selected) clearHighlight(); })
      .on('click', (e,d) => {
        e.stopPropagation();
        if (selected === d.id) { clearHighlight(); return; }
        selected = d.id;
        node.classed('selected', n => n.id === d.id);
        highlight(d);
      });

  nodes.forEach(n => {
    if (n._isNew || n._shouldSpike) {
      const el = gNodes.select(\`#node-\${n.id}\`).select('circle');
      if (el.empty()) return;
      const baseR = nodeRadius(n);
      const color = getNodeColor(n);
      const stroke = getUserStroke(n);
      el.interrupt()
        .attr('r', baseR * 1.5)
        .attr('fill', color)
        .attr('stroke', '#ffffff').attr('stroke-width', 6).attr('filter', 'url(#glow-white)')
        .transition().duration(n._isNew ? 1000 : 600).ease(d3.easeQuadOut)
        .attr('r', baseR)
        .attr('fill', color + '44')
        .attr('stroke', 'none').attr('stroke-width', 0)
        .attr('filter', n.visibility === 'shared' ? 'url(#glow-emerald)' : (
          () => {
             const uid = n.user_id;
             const paletteIndex = Array.from(new Set(nodes.map(node => node.user_id))).indexOf(uid);
             const glowNames = ['blue','purple','orange','green','white','emerald'];
             return \`url(#glow-\${glowNames[paletteIndex % glowNames.length]})\`;
          }
        )());
      // Pulse connected links
      gLinks.selectAll('.link')
        .filter(l => (l.source.id || l.source) === n.id || (l.target.id || l.target) === n.id)
        .interrupt()
        .classed('firing', true)
        .transition().duration(n._isNew ? 1200 : 800)
        .on('end', function() { d3.select(this).classed('firing', false); });

      n._isNew = false; n._shouldSpike = false;
    }
  });

  sim.nodes(nodes);
  sim.force('link').links(links);
  sim.alpha(0.05).restart();
}

sim.on('tick', () => {
  if (!link || !node) return;
  link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
      .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  node.attr('transform', d => \`translate(\${d.x},\${d.y})\`);
});

svg.on('click', clearHighlight);
document.getElementById('search-input').addEventListener('input', e => {
  const q = e.target.value.toLowerCase().trim();
  if (!q) { 
    node.classed('dimmed', false); 
    link.classed('dimmed', false); 
    return; 
  }
  
  let firstMatch = null;
  node.classed('dimmed', n => {
    const match = n.label.toLowerCase().includes(q);
    if (match && !firstMatch) firstMatch = n;
    return !match;
  });
  link.classed('dimmed', true);

  if (firstMatch) {
    const transform = d3.zoomIdentity
      .translate(W/2, H/2)
      .scale(1.2)
      .translate(-firstMatch.x, -firstMatch.y);
    svg.transition().duration(750).call(zoom.transform, transform);
    highlight(firstMatch);
  }
});

update();
setInterval(update, 2000);
</script>
</body>
</html>`;
}


const server = http.createServer(async (req, res) => {
  const url = req.url ?? "/";

  if (req.method === "GET" && url === "/") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(renderBrain());
    return;
  }

  if (req.method === "GET" && url === "/api/graph") {
    json(res, await buildGraphData());
    return;
  }

  if (req.method === "POST") {
    const body = await readBody(req);

    if (url === "/delete-node") {
      try { await deleteNode(body.id); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/delete-edge") {
      try { await deleteEdge(body.from_id, body.to_id); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/rename") {
      try { await renameNode(body.id, body.label); json(res, { ok: true }); }
      catch (e: any) { json(res, { ok: false, error: e.message }); }
      return;
    }

    if (url === "/merge") {
      try {
        await rewireEdges(body.delete_id, body.keep_id);
        await deleteNode(body.delete_id);
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
