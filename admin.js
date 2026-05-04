// ─── State ────────────────────────────────────────────────────────────────────
let network      = null;
let edgesDataSet = null;
let nodesDataSet = null;
let currentPath  = [];
let animSpeed    = 750;
let liveSpeed    = 750;
let lastData     = null;
let liveSource   = null;
let liveDebounce = false;

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
        dot.className    = 'status-dot';
        text.textContent = `v${d.version} · ${d.device?.toUpperCase()} · ${d.uptime}`;
        const devBadge = document.getElementById('device-badge');
        if (devBadge) { devBadge.style.display = ''; devBadge.textContent = d.device?.toUpperCase() || 'CPU'; }
    } catch {
        dot.className    = 'status-dot offline';
        text.textContent = 'Backend offline';
    }
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${id}`).classList.add('active');
        });
    });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    setInterval(checkStatus, 30000);
    initTabs();

    const saved = localStorage.getItem('admin_last_result');
    if (saved) {
        try { lastData = JSON.parse(saved); renderDashboard(lastData); } catch (_) {}
    }

    document.getElementById('speed-slider').addEventListener('input', e => {
        animSpeed = parseInt(e.target.value);
        document.getElementById('speed-label').textContent = animSpeed + 'ms';
    });
    document.getElementById('live-speed-slider').addEventListener('input', e => {
        liveSpeed = parseInt(e.target.value);
        document.getElementById('live-speed-label').textContent = liveSpeed + 'ms';
    });

    document.getElementById('btn-export-json').addEventListener('click', exportJSON);
    document.getElementById('btn-export-csv').addEventListener('click', exportCSV);
    document.getElementById('btn-live').addEventListener('click', toggleLive);
    document.getElementById('btn-retrain').addEventListener('click', doRetrain);
    document.getElementById('btn-replay').addEventListener('click', () => animatePath(currentPath));
});

// ─── Simulate ─────────────────────────────────────────────────────────────────
document.getElementById('sim-form').addEventListener('submit', async e => {
    e.preventDefault();
    stopLive();

    const btn  = document.getElementById('btn-generate');
    const orig = btn.innerHTML;
    btn.innerHTML = '<span class="spinner"></span> Running…';
    btn.disabled  = true;

    try {
        const res = await fetch('/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode:            'admin',
                nodes:           parseInt(document.getElementById('node-count').value),
                malicious_ratio: parseFloat(document.getElementById('mal-ratio').value),
                test_episodes:   parseInt(document.getElementById('test-episodes').value) || 100,
                topology:        document.getElementById('topology-select').value,
                multi_path:      document.getElementById('multi-path').checked,
            }),
        });
        if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error || `HTTP ${res.status}`); }
        lastData = await res.json();
        localStorage.setItem('admin_last_result', JSON.stringify(lastData));
        renderDashboard(lastData);
        toast('success', 'Simulation complete',
            `Accuracy ${lastData.accuracy}%  ·  Crash rate ${lastData.crash_rate}%`);
    } catch (err) {
        toast('error', 'Simulation failed', err.message);
    } finally {
        btn.innerHTML = orig;
        btn.disabled  = false;
    }
});

// ─── Live ─────────────────────────────────────────────────────────────────────
function toggleLive() {
    if (liveSource) { stopLive(); return; }
    if (liveDebounce) return;
    liveDebounce = true;
    setTimeout(() => { liveDebounce = false; }, 600);

    const btn      = document.getElementById('btn-live');
    const nodes    = document.getElementById('node-count')?.value || 20;
    const malRatio = document.getElementById('mal-ratio')?.value  || 0.2;
    const topology = document.getElementById('topology-select')?.value || '';

    btn.innerHTML = '<span class="spinner"></span> Stop stream';
    document.getElementById('live-badge').classList.add('visible');

    liveSource = new EventSource(
        `/stream_route?nodes=${nodes}&mal_ratio=${malRatio}&topology=${encodeURIComponent(topology)}&speed_ms=${liveSpeed}`
    );

    liveSource.onmessage = e => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'init') {
            document.getElementById('empty-state')?.remove();
            renderGraph(msg.nodes, msg.edges);
            currentPath = [msg.source];
            hideMuliPathLegend();
            updateStats(msg.node_count, msg.edge_count, msg.malicious_count);
            setBadge('Routing', '');
        } else if (msg.type === 'step') {
            highlightLiveEdge(msg.from, msg.to);
        } else if (msg.type === 'done') {
            setBadge(msg.reason, msg.reason);
            setHops(msg.hop_count);
            currentPath = msg.path;
            document.getElementById('btn-replay').disabled = false;
            stopLive();
            const t = msg.reason === 'SUCCESS' ? 'success' : msg.reason === 'MALICIOUS' ? 'error' : 'warning';
            toast(t, `Route ${msg.reason.toLowerCase()}`, `${msg.hop_count} hops`);
        }
    };
    liveSource.onerror = () => { stopLive(); toast('error', 'Stream disconnected', ''); };
}

function stopLive() {
    if (liveSource) { liveSource.close(); liveSource = null; }
    const btn = document.getElementById('btn-live');
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg> Start Live Route`;
    document.getElementById('live-badge').classList.remove('visible');
}

// ─── Render ───────────────────────────────────────────────────────────────────
function renderDashboard(data) {
    document.getElementById('empty-state')?.remove();
    document.getElementById('val-accuracy').textContent = data.accuracy != null ? data.accuracy + '%' : '—';
    document.getElementById('val-crash').textContent    = data.crash_rate != null ? data.crash_rate + '%' : '—';
    setBadge(data.reason, data.reason);
    setHops(data.hop_count ?? (data.demo_path?.length - 1));
    renderGraph(data.nodes, data.edges);
    updateStats(data.node_count, data.edge_count, data.malicious_count);
    document.getElementById('btn-replay').disabled = false;
    document.getElementById('btn-export-json').disabled = false;
    document.getElementById('btn-export-csv').disabled  = false;

    currentPath = data.demo_path;
    const hasAlt = data.alt_paths?.length > 0;
    if (hasAlt) {
        const all = [data.demo_path, ...data.alt_paths.map(p => p.path)];
        showMultiPathLegend(all.length);
        setTimeout(() => all.forEach((p, i) => animateSinglePath(p, PATH_COLORS[i], i * 500)), 400);
    } else {
        hideMuliPathLegend();
        setTimeout(() => { network?.fit({ animation: { duration: 500 } }); animatePath(currentPath); }, 300);
    }
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
    const container = document.getElementById('mynetwork');
    if (network) network.destroy();
    network = new vis.Network(container, { nodes: nodesDataSet, edges: edgesDataSet },
        { physics: { enabled: false }, interaction: { hover: true, tooltipDelay: 100, zoomView: true } });
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
        _highlightEdge(pathArray[step], pathArray[step + 1], PATH_COLORS[0], all);
        network?.focus(pathArray[step + 1], {
            scale: 1.5, animation: { duration: Math.round(animSpeed * 0.7), easingFunction: 'easeInOutQuad' }
        });
        step++;
    }, animSpeed);
}

function animateSinglePath(pathArray, color, delayMs) {
    if (!pathArray || pathArray.length < 2) return;
    const all = edgesDataSet.get();
    setTimeout(() => {
        let step = 0;
        const go = setInterval(() => {
            if (step >= pathArray.length - 1) { clearInterval(go); return; }
            _highlightEdge(pathArray[step], pathArray[step + 1], color, all);
            step++;
        }, animSpeed);
    }, delayMs);
}

function _highlightEdge(u, v, color, all) {
    const re = new RegExp(`^${u}-${v}$|^${v}-${u}$`);
    const e  = all.find(x => re.test(x.id));
    if (e) edgesDataSet.update({ id: e.id, color: { color }, width: 3 });
}

function highlightLiveEdge(from, to) {
    if (!edgesDataSet) return;
    _highlightEdge(from, to, PATH_COLORS[0], edgesDataSet.get());
    network?.focus(to, { scale: 1.5, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
}

// ─── Multi-path Legend ────────────────────────────────────────────────────────
function showMultiPathLegend(count) {
    const leg = document.getElementById('path-legend');
    leg.style.display = 'flex';
    leg.innerHTML = PATH_COLORS.slice(0, count).map((c, i) =>
        `<div class="path-legend-item"><div class="path-swatch" style="background:${c}"></div>Path ${i + 1}</div>`
    ).join('');
}

function hideMuliPathLegend() {
    const leg = document.getElementById('path-legend');
    if (leg) { leg.style.display = 'none'; leg.innerHTML = ''; }
}

// ─── UI Helpers ───────────────────────────────────────────────────────────────
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

// ─── Retrain ──────────────────────────────────────────────────────────────────
async function doRetrain() {
    const episodes = parseInt(document.getElementById('retrain-episodes').value) || 500;
    const btn      = document.getElementById('btn-retrain');
    const statusEl = document.getElementById('retrain-status');
    const msg      = document.getElementById('retrain-msg');
    const bar      = document.getElementById('retrain-bar');
    const pct      = document.getElementById('retrain-pct');

    btn.innerHTML = '<span class="spinner"></span> Training…';
    btn.disabled  = true;
    statusEl.style.display = '';
    msg.textContent = `${episodes} episodes queued`;
    bar.style.width = '4%';
    pct.textContent = '4%';

    try {
        const res = await fetch('/retrain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ episodes }),
        });
        if (res.status === 409) {
            toast('warning', 'Retrain in progress', 'Wait for the current run to finish.');
            btn.disabled = false;
            btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg> Start Retraining`;
            return;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
        toast('error', 'Retrain failed', err.message);
        btn.disabled = false;
        btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg> Start Retraining`;
        return;
    }

    // Poll /retrain/status for real progress
    const poll = setInterval(async () => {
        try {
            const s = await fetch('/retrain/status').then(r => r.json());
            const p = Math.max(4, s.pct);
            bar.style.width = p + '%'; pct.textContent = p + '%';
            msg.textContent = s.done ? (s.success ? 'Complete' : 'Failed') : `${s.progress} / ${s.total} episodes`;
            if (s.done) {
                clearInterval(poll);
                bar.style.width = '100%'; pct.textContent = '100%';
                if (s.success) {
                    toast('success', 'Retrain complete', `${s.episodes} episodes · weights saved`);
                    setTimeout(() => { statusEl.style.display = 'none'; bar.style.width = '0%'; }, 3000);
                    checkStatus();
                } else {
                    toast('error', 'Retrain failed', 'Check server logs.');
                }
                btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg> Start Retraining`;
                btn.disabled = false;
            }
        } catch (_) {}
    }, 1000);
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
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: name });
    a.click(); URL.revokeObjectURL(a.href);
}
