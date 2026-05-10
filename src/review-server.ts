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


function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

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
    isHub: n.label.startsWith("concept:"),
    contextName: n.label.startsWith("[ctx:") ? n.label.slice(5, -1) : null,
    contexts: ctxMap.get(n.id) ?? [],
    strength: n.strength,
    importance: n.importance,
    access_count: n.access_count,
    last_accessed_at: n.last_accessed_at,
    last_fired_at: n.last_fired_at,
    metadata: n.metadata,
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
  <span class="title">MEMORY BRAIN</span>
</div>

<div id="info-panel">
  <div class="label" id="info-label"></div>
  <div class="row"><span class="key">Strength</span><span class="val" id="info-strength"></span></div>
  <div class="row"><span class="key">Importance</span><span class="val" id="info-importance"></span></div>
  <div class="row"><span class="key">Accesses</span><span class="val" id="info-access"></span></div>
  <div class="row"><span class="key">Last seen</span><span class="val" id="info-last"></span></div>
  <div style="margin-top:8px; color:#555; font-size:0.72rem">Contexts / Types</div>
  <div id="info-contexts" style="margin-top:4px"></div>
</div>

<div id="legend">
  <div class="row"><div class="swatch" style="background:#8b5cf6"></div><span>Context hub</span></div>
  <div class="row"><div class="swatch" style="background:#fbbf24"></div><span>Conceptual Hub</span></div>
  <div class="row"><div class="swatch" style="background:#38bdf8"></div><span>user context</span></div>
  <div id="legend-projects"></div>
  <div class="row"><div class="swatch" style="background:#94a3b8"></div><span>no context</span></div>
  <div class="row"><div class="swatch" style="background:#ef4444"></div><span>Conflict (Resolution needed)</span></div>
  <div class="row"><div class="swatch" style="background:#f97316"></div><span>Curiosity (Gap detected)</span></div>
  <div style="margin-top:8px; border-top:1px solid #1e2a3a; padding-top:8px">
    <div class="row"><div class="line-swatch" style="background:#fbbf24"></div><span>abstraction</span></div>
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
defs.append('filter').attr('id', 'glow-hub').attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%')
  .append('feGaussianBlur').attr('stdDeviation','6').attr('result','blur');
d3.select('#glow-hub').append('feMerge').selectAll('feMergeNode').data(['blur','SourceGraphic']).join('feMergeNode').attr('in',d=>d);

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
    if (n.label.startsWith('conflict:')) return '#ef4444';
    if (n.label.startsWith('curiosity:')) return '#f97316';
    if (n.isContext) return '#8b5cf6';
    if (n.isHub) return '#fbbf24';
    if (n.contexts.length > 0) {
      const c = n.contexts.find(c => c !== 'user') ?? n.contexts[0];
      return ctxColor[c] ?? '#94a3b8';
    }
    return '#94a3b8';
  }

  function nodeRadius(n) {
    if (n.isContext) return 14 + n.access_count * 0.4;
    if (n.isHub) return 16 + n.importance * 10;
    return 5 + n.strength * 14 + Math.min(n.access_count, 20) * 0.3;
  }

  function linkColor(type) {
    return { abstraction:'#fbbf24', causal:'#f97316', semantic:'#38bdf8', temporal:'#4ade80', episodic:'#c084fc' }[type] ?? '#888';
  }

  // Build id sets for fast lookup
  const nodeById = new Map(nodes.map(n => [n.id, n]));

  // Simulation
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(d => {
      const s = nodeById.get(d.source.id ?? d.source);
      const t = nodeById.get(d.target.id ?? d.target);
      if (s?.isContext || t?.isContext) return 80;
      if (s?.isHub || t?.isHub) return 60;
      return 120 / (d.weight + 0.1);
    }).strength(d => d.weight * 0.4))
    .force('charge', d3.forceManyBody().strength(d => (d.isContext || d.isHub) ? -500 : -120))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 10));

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
    .attr('id', d => \`node-\${d.id}\`)
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; })
    );

  node.append('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => nodeColor(d) + (d.isContext || d.isHub ? '22' : '18'))
    .attr('stroke', d => nodeColor(d))
    .attr('stroke-width', d => d.isContext || d.isHub || d.label.startsWith('conflict:') ? 2 : 1.5)
    .attr('filter', d => {
      if (d.label.startsWith('conflict:')) return 'url(#glow-orange)';
      if (d.isHub) return 'url(#glow-hub)';
      if (d.isContext) return 'url(#glow-purple)';
      return 'url(#glow-blue)';
    });

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
    document.getElementById('info-label').textContent = d.isContext ? d.contextName : (d.isHub ? d.label.replace('concept:','') : d.label);
    document.getElementById('info-strength').textContent = d.strength.toFixed(3);
    document.getElementById('info-importance').textContent = (d.importance || 0.5).toFixed(3);
    document.getElementById('info-access').textContent = d.access_count;
    document.getElementById('info-last').textContent = new Date(d.last_accessed_at).toLocaleDateString();
    document.getElementById('info-contexts').innerHTML =
      d.isContext
        ? '<span class="ctx-tag">context hub</span>'
        : (d.isHub ? '<span class="ctx-tag" style="background:#3a2a0a;color:#fbbf24">conceptual hub</span>' : '') + 
          (d.contexts.length ? d.contexts.map(c => '<span class="ctx-tag">' + c + '</span>').join('') : (!d.isHub ? '<span style="color:#444">none</span>' : ''));
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

  // Real-time polling for updates
  setInterval(() => {
    fetch('/api/graph').then(r => r.json()).then(data => {
      data.nodes.forEach(nn => {
        const existing = nodeById.get(nn.id);
        if (existing && nn.last_fired_at > existing.last_fired_at) {
          console.log('Spike:', nn.label);
          existing.last_fired_at = nn.last_fired_at;
          existing.strength = nn.strength;
          existing.importance = nn.importance;
               // Trigger Neuronal Spike (Refined)
          const el = node.filter(d => d.id === nn.id).select('circle');
          const baseR = nodeRadius(existing);
          const color = nodeColor(existing);
          const baseIntensity = (nn.strength * 0.8) + 0.2;
          
          // Staggered firing: small delay based on ID
          const idShift = (parseInt(nn.id.slice(0, 8), 16) % 1000) / 1000;
          
          setTimeout(() => {
            el.interrupt()
              .attr('r', baseR * (1.2 + baseIntensity * 0.8))
              .attr('stroke', '#ffffff')
              .attr('stroke-width', 2 + baseIntensity * 12)
              .attr('filter', 'url(#glow-white)')
              .transition().duration(4000).ease(d3.easeQuadOut)
              .attr('r', baseR)
              .attr('stroke', color)
              .attr('stroke-width', existing.isContext || existing.isHub ? 2 : 1.5)
              .attr('filter', existing.isHub ? 'url(#glow-hub)' : (existing.isContext ? 'url(#glow-purple)' : 'url(#glow-blue)'));
          }, idShift * 600);
          
          existing.last_fired_at = nn.last_fired_at;
          existing.strength = nn.strength;
          existing.importance = nn.importance;
        }
      });
    });
  }, 1000);

  // Animate pulse on context nodes
  function pulseContexts() {
    container.selectAll('circle[stroke-dasharray]')
      .transition().duration(2000).ease(d3.easeSinInOut)
      .attr('r', function() { 
        const d = d3.select(this.parentNode).datum();
        return parseFloat(d.isContext ? nodeRadius(d) + 12 : 0); 
      })
      .attr('opacity', 0.05)
      .transition().duration(2000).ease(d3.easeSinInOut)
      .attr('r', function() { 
        const d = d3.select(this.parentNode).datum();
        return parseFloat(d.isContext ? nodeRadius(d) + 6 : 0); 
      })
      .attr('opacity', 0.3)
      .on('end', pulseContexts);
  }
  pulseContexts();
});
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
