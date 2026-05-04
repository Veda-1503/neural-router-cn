// ─── State ────────────────────────────────────────────────────────────────────
let network      = null;
let edgesDataSet = null;
let nodesDataSet = null;
let currentPath  = [];
let animSpeed    = 750;
let lastData     = null;
let liveSource   = null;

const PATH_COLORS = ['#8b5cf6', '#06b6d4', '#f97316'];

// ─── Toast ────────────────────────────────────────────────────────────────────
function toast(type, title, msg, duration = 4000) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `
        <div class="toast-indicator"></div>
        <div class="toast-body">
            <div class="toast-title">${title}</div>
            ${msg ? `<div class="toast-msg">${msg}</div>` : ''}
        </div>`;
    document.getElementById('toast-root').appendChild(el);
    setTimeout(() => {
        el.classList.add('out');
        el.addEventListener('animationend', () => el.remove(), { once: true });
    }, duration);
}

// ─── Status ───────────────────────────────────────────────────────────────────
async function checkStatus() {
    const dot  = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    try {
        const r = await fetch('/api/status');
        if (!r.ok) throw new Error();
        const d = await r.json();
        dot.className  = 'status-dot';
        text.textContent = `v${d.version} · ${d.device?.toUpperCase()} · ${d.uptime}`;
    } catch {
        dot.className  = 'status-dot offline';
        text.textContent = 'Backend offline';
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    setInterval(checkStatus, 30000);

    document.getElementById('speed-slider').addEventListener('input', e => {
        animSpeed = parseInt(e.target.value);
        document.getElementById('speed-label').textContent = animSpeed + 'ms';
    });

    document.getElementById('btn-export-json').addEventListener('click', exportJSON);
    document.getElementById('btn-export-csv').addEventListener('click', exportCSV);
    document.getElementById('btn-deploy').addEventListener('click', deployPacket);
    document.getElementById('btn-live').addEventListener('click', toggleLive);
    document.getElementById('btn-replay').addEventListener('click', () => animatePath(currentPath));
});

// ─── Deploy ───────────────────────────────────────────────────────────────────
async function deployPacket() {
    stopLive();
    const btn  = document.getElementById('btn-deploy');
    const orig = btn.innerHTML;
    btn.innerHTML = '<span class="spinner"></span> Routing…';
    btn.disabled  = true;

    try {
        const res = await fetch('/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'user' }),
        });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        lastData = await res.json();
        renderDashboard(lastData);
        toast('success', 'Route complete',
            `${lastData.hop_count} hop${lastData.hop_count !== 1 ? 's' : ''} · ${lastData.reason}`);
    } catch (err) {
        toast('error', 'Request failed', err.message);
    } finally {
        btn.innerHTML = orig;
        btn.disabled  = false;
    }
}

// ─── Live / SSE ───────────────────────────────────────────────────────────────
function toggleLive() {
    if (liveSource) { stopLive(); return; }

    const btn = document.getElementById('btn-live');
    btn.innerHTML = '<span class="spinner"></span> Stop stream';
    document.getElementById('btn-deploy').disabled = true;
    document.getElementById('live-badge').classList.add('visible');

    liveSource = new EventSource(
        `/stream_route?nodes=20&mal_ratio=0.2&topology=&speed_ms=${animSpeed}`
    );

    liveSource.onmessage = e => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'init') {
            document.getElementById('empty-state')?.remove();
            renderGraph(msg.nodes, msg.edges);
            currentPath = [msg.source];
            updateStats(msg.node_count, msg.edge_count, msg.malicious_count);
            setBadge('Routing', '');
        } else if (msg.type === 'step') {
            highlightLiveEdge(msg.from, msg.to);
        } else if (msg.type === 'done') {
            setBadge(msg.reason, msg.reason);
            setHops(msg.hop_count);
            currentPath = msg.path;
            renderPathFlow(msg.path, msg.reason);
            document.getElementById('btn-replay').disabled = false;
            enableExport();
            stopLive();
            const t = msg.reason === 'SUCCESS' ? 'success' : msg.reason === 'MALICIOUS' ? 'error' : 'warning';
            toast(t, `Route ${msg.reason.toLowerCase()}`, `${msg.hop_count} hops`);
        }
    };

    liveSource.onerror = () => {
        stopLive();
        toast('error', 'Stream disconnected', 'SSE connection lost.');
    };
}

function stopLive() {
    if (liveSource) { liveSource.close(); liveSource = null; }
    const btn = document.getElementById('btn-live');
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg> Stream Live Route`;
    document.getElementById('btn-deploy').disabled = false;
    document.getElementById('live-badge').classList.remove('visible');
}

// ─── Render ───────────────────────────────────────────────────────────────────
function renderDashboard(data) {
    document.getElementById('empty-state')?.remove();
    setBadge(data.reason, data.reason);
    setHops(data.hop_count ?? (data.demo_path?.length - 1));
    currentPath = data.demo_path;
    renderGraph(data.nodes, data.edges);
    updateStats(data.node_count, data.edge_count, data.malicious_count);
    renderPathFlow(data.demo_path, data.reason);
    document.getElementById('btn-replay').disabled = false;
    enableExport();
    setTimeout(() => { network?.fit({ animation: { duration: 500 } }); animatePath(currentPath); }, 150);
}

// ─── Graph ────────────────────────────────────────────────────────────────────
function renderGraph(nodes, edges) {
    const visNodes = nodes.map(n => {
        let bg = '#3b82f6', border = '#2563eb';
        if (n.group === 'malicious') { bg = '#ef4444'; border = '#dc2626'; }
        if (n.is_source)             { bg = '#22c55e'; border = '#16a34a'; }
        if (n.is_dest)               { bg = '#f59e0b'; border = '#d97706'; }
        const role = n.is_source ? 'Origin' : n.is_dest ? 'Destination'
                   : n.group === 'malicious' ? 'Compromised' : 'Secure';
        return {
            id: n.id, label: String(n.id),
            title: `Node ${n.id}  |  Trust ${n.trust !== undefined ? n.trust.toFixed(2) : 'n/a'}  |  ${role}`,
            x: n.x, y: n.y,
            color: { background: bg, border, highlight: { background: bg, border: '#ffffff' } },
            font: { color: '#fff', size: 11, face: 'JetBrains Mono, monospace', bold: false },
            shape: 'dot', size: n.is_source || n.is_dest ? 20 : 10,
            borderWidth: 1.5,
        };
    });
    const visEdges = edges.map(e => ({
        id: `${e.from}-${e.to}`, from: e.from, to: e.to,
        color: { color: '#252525', highlight: '#333333' },
        width: 1, smooth: { type: 'continuous' },
    }));
    nodesDataSet = new vis.DataSet(visNodes);
    edgesDataSet = new vis.DataSet(visEdges);
    if (network) network.destroy();
    network = new vis.Network(
        document.getElementById('mynetwork'),
        { nodes: nodesDataSet, edges: edgesDataSet },
        { physics: { enabled: false }, interaction: { hover: true, tooltipDelay: 100, zoomView: true } }
    );
}

// ─── Animation ────────────────────────────────────────────────────────────────
function animatePath(pathArray) {
    if (!pathArray || pathArray.length < 2 || !edgesDataSet) return;
    const all = edgesDataSet.get();
    edgesDataSet.update(all.map(e => ({ id: e.id, color: { color: '#1c1c1c' }, width: 1 })));
    network?.fit({ animation: { duration: 500 } });
    let step = 0;
    const go = setInterval(() => {
        if (step >= pathArray.length - 1) { clearInterval(go); return; }
        _highlightEdge(pathArray[step], pathArray[step + 1], '#8b5cf6', all);
        network?.focus(pathArray[step + 1], {
            scale: 1.5, animation: { duration: Math.round(animSpeed * 0.7), easingFunction: 'easeInOutQuad' }
        });
        step++;
    }, animSpeed);
}

function _highlightEdge(u, v, color, all) {
    const re = new RegExp(`^${u}-${v}$|^${v}-${u}$`);
    const e  = all.find(x => re.test(x.id));
    if (e) edgesDataSet.update({ id: e.id, color: { color }, width: 3 });
}

function highlightLiveEdge(from, to) {
    if (!edgesDataSet) return;
    _highlightEdge(from, to, '#8b5cf6', edgesDataSet.get());
    network?.focus(to, { scale: 1.5, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
}

// ─── UI helpers ───────────────────────────────────────────────────────────────
function setBadge(text, cls) {
    const b = document.getElementById('val-reason');
    b.textContent = text; b.className = 'badge' + (cls ? ' ' + cls : '');
}

function setHops(n) {
    const el = document.getElementById('val-hops');
    el.textContent = `${n} hop${n !== 1 ? 's' : ''}`;
    el.style.display = '';
}

function updateStats(nodes, edges, mal) {
    if (nodes == null) return;
    document.getElementById('topo-stats').style.display = 'flex';
    document.getElementById('stat-nodes').textContent = nodes;
    document.getElementById('stat-edges').textContent = edges;
    document.getElementById('stat-mal').textContent   = mal;
}

function renderPathFlow(path, reason) {
    if (!path?.length) return;
    const panel = document.getElementById('route-panel');
    const flow  = document.getElementById('path-flow');
    panel.style.display = '';
    const isMalicious = String(reason).toUpperCase() === 'MALICIOUS';
    flow.innerHTML = path.map((node, i) => {
        let cls = 'path-node';
        if (i === 0) cls += ' src';
        else if (i === path.length - 1 && !isMalicious) cls += ' dst';
        else if (i === path.length - 1 && isMalicious)  cls += ' threat';
        const arrow = i < path.length - 1
            ? '<span class="path-arrow">›</span>' : '';
        return `<span class="${cls}">${node}</span>${arrow}`;
    }).join('');
}

function enableExport() {
    document.getElementById('btn-export-json').disabled = false;
    document.getElementById('btn-export-csv').disabled  = false;
}

// ─── Export ───────────────────────────────────────────────────────────────────
function exportJSON() {
    if (!lastData) return;
    _dl(new Blob([JSON.stringify(lastData, null, 2)], { type: 'application/json' }), 'routing_result.json');
    toast('info', 'Exported', 'routing_result.json');
}

function exportCSV() {
    if (!lastData?.nodes) return;
    const h = ['node_id', 'group', 'trust', 'x', 'y', 'is_source', 'is_dest'];
    const r = lastData.nodes.map(n =>
        [n.id, n.group, n.trust?.toFixed(2) ?? '', n.x?.toFixed(2), n.y?.toFixed(2), n.is_source, n.is_dest]);
    _dl(new Blob([[h, ...r].map(x => x.join(',')).join('\n')], { type: 'text/csv' }), 'routing_nodes.csv');
    toast('info', 'Exported', 'routing_nodes.csv');
}

function _dl(blob, name) {
    const a = Object.assign(document.createElement('a'),
              { href: URL.createObjectURL(blob), download: name });
    a.click();
    URL.revokeObjectURL(a.href);
}
