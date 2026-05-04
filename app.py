from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import torch
import os
import random
import json
import time
import logging
import numpy as np
from datetime import datetime

from config import (NUM_NODES, MALICIOUS_RATIO, FEATURE_SIZE, MODEL_PATH,
                    HOST, PORT, APP_NAME, APP_VERSION)
from network_env import NetworkEnvironment
from neural_router import NeuralRouterAgent

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(APP_NAME)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

_start_time = datetime.utcnow()
_retrain_lock     = threading.Lock()  # Thread-safe lock to prevent concurrent retrains
_retrain_status   = {"running": False, "progress": 0, "total": 0, "done": False, "success": None, "episodes": 0}

# ─── Security Headers ─────────────────────────────────────────────────────────

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options']        = 'SAMEORIGIN'
    response.headers['Referrer-Policy']        = 'strict-origin-when-cross-origin'
    return response

# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "code": 404}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "code": 405}), 405

@app.errorhandler(500)
def internal_error(e):
    log.exception("Internal server error")
    return jsonify({"error": "Internal server error", "code": 500}), 500

# ─── Model Init ───────────────────────────────────────────────────────────────

log.info("Initializing %s v%s...", APP_NAME, APP_VERSION)
agent = NeuralRouterAgent(feature_size=FEATURE_SIZE)
env   = NetworkEnvironment(num_nodes=NUM_NODES, malicious_ratio=MALICIOUS_RATIO, randomize_graph=True)


def _load_model():
    """Load saved weights with architecture compatibility check."""
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        expected = sum(p.numel() for p in agent.model.parameters())
        saved    = sum(v.numel() for v in state_dict.values())
        if expected != saved:
            log.warning("Param count mismatch (%d saved vs %d current). Re-training.", saved, expected)
            return False
        agent.model.load_state_dict(state_dict)
        agent.model.eval()
        log.info("Model loaded from %s", MODEL_PATH)
        return True
    except Exception as e:
        log.warning("Failed to load model: %s. Re-training.", e)
        return False


def _train_model(episodes=1500):
    agent.epsilon = 1.0
    for ep in range(episodes):
        valid_actions, features = env.reset(episode=ep)
        done = False
        while not done:
            action, action_feature = agent.act(valid_actions, features)
            if action is None:
                break
            (next_actions, next_features), reward, done, info = env.step(action)
            agent.remember(action_feature, reward, next_features, done)
            agent.replay()
            valid_actions, features = next_actions, next_features
    torch.save(agent.model.state_dict(), MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)


if os.path.exists(MODEL_PATH):
    log.info("Loading pre-trained model weights...")
    if not _load_model():
        log.info("Re-training model (%d eps)...", 1500)
        _train_model(1500)
else:
    log.info("No weights found — training from scratch (1500 eps)...")
    _train_model(1500)

agent.epsilon = 0.0
agent.model.eval()
log.info("Agent ready — epsilon=0.0, model in eval mode.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_topology(raw: str):
    """Convert the frontend topology string to a valid value for NetworkEnvironment."""
    VALID = {'barabasi_albert', 'erdos_renyi', 'watts_strogatz'}
    return raw if raw in VALID else None


def _build_nodes_edges(custom_env, demo_src, demo_dst):
    nodes_data, edges_data = [], []
    for n in custom_env.graph.nodes():
        nodes_data.append({
            "id":        n,
            "label":     f"{n}\n{custom_env.graph.nodes[n]['trust']:.2f}",
            "trust":     float(custom_env.graph.nodes[n]['trust']),
            "x":         float(custom_env.graph.nodes[n]['pos'][0]) * 600,
            "y":         float(custom_env.graph.nodes[n]['pos'][1]) * 600,
            "group":     "malicious" if n in custom_env.malicious_nodes else "safe",
            "is_source": n == demo_src,
            "is_dest":   n == demo_dst,
        })
    for u, v in custom_env.graph.edges():
        edges_data.append({"from": u, "to": v})
    return nodes_data, edges_data


def _run_single_path(custom_env, src, dst):
    """Run a single agent episode from src→dst on the current graph. Returns path + reason."""
    custom_env.source       = src
    custom_env.destination  = dst
    custom_env.current_node = src
    custom_env.visited      = {src}
    custom_env.steps        = 0
    valid_actions, features = custom_env.get_state()
    done = False
    path = [src]
    info = {"reason": "step"}
    while not done:
        action, _ = agent.act(valid_actions, features)
        if action is None:
            break
        (valid_actions, features), _, done, info = custom_env.step(action)
        path.append(custom_env.current_node)
    return path, info["reason"].upper()


# ─── Static Routes ────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/admin')
def admin():
    return app.send_static_file('admin.html')


# ─── Health & Status ──────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)})


@app.route('/api/status')
def api_status():
    uptime_s = int((datetime.utcnow() - _start_time).total_seconds())
    h, rem   = divmod(uptime_s, 3600)
    m, s     = divmod(rem, 60)
    return jsonify({
        "app":          APP_NAME,
        "version":      APP_VERSION,
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path":   MODEL_PATH,
        "device":       str(agent.device),
        "uptime":       f"{h:02d}:{m:02d}:{s:02d}",
        "uptime_s":     uptime_s,
        "epsilon":      round(agent.epsilon, 4),
        "replay_size":  len(agent.memory),
    })


# ─── Retrain ──────────────────────────────────────────────────────────────────

@app.route('/retrain', methods=['POST'])
def retrain():
    if not _retrain_lock.acquire(blocking=False):
        return jsonify({"error": "Retrain already in progress"}), 409

    data = request.json or {}
    try:
        episodes = max(100, min(int(data.get('episodes', 500)), 5000))
    except (ValueError, TypeError):
        _retrain_lock.release()
        return jsonify({"error": "episodes must be an integer between 100 and 5000"}), 400

    def _bg_train():
        global _retrain_status
        _retrain_status.update({"running": True, "progress": 0, "total": episodes,
                                 "done": False, "success": None, "episodes": episodes})
        log.info("Background retrain started: %d episodes", episodes)
        try:
            agent.epsilon = 1.0
            for ep in range(episodes):
                valid_actions, features = env.reset(episode=ep)
                done = False
                while not done:
                    action, action_feature = agent.act(valid_actions, features)
                    if action is None:
                        break
                    (valid_actions, features), reward, done, _ = env.step(action)
                    agent.remember(action_feature, reward, valid_actions, done)
                    agent.replay()
                _retrain_status["progress"] = ep + 1
            torch.save(agent.model.state_dict(), MODEL_PATH)
            agent.epsilon = 0.0
            agent.model.eval()
            _retrain_status.update({"running": False, "done": True, "success": True})
            log.info("Background retrain complete.")
        except Exception as exc:
            log.exception("Background retrain failed: %s", exc)
            _retrain_status.update({"running": False, "done": True, "success": False})
        finally:
            _retrain_lock.release()

    threading.Thread(target=_bg_train, daemon=True).start()
    return jsonify({"status": "started", "episodes": episodes}), 202


@app.route('/retrain/status')
def retrain_status():
    s = _retrain_status
    pct = int((s["progress"] / s["total"]) * 100) if s["total"] else 0
    return jsonify({
        "running":  s["running"],
        "done":     s["done"],
        "success":  s["success"],
        "progress": s["progress"],
        "total":    s["total"],
        "pct":      pct,
        "episodes": s["episodes"],
    })


# ─── Simulate (batch) ─────────────────────────────────────────────────────────

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    mode = data.get('mode', 'simulation')

    if mode == 'user':
        nodes     = random.randint(15, 30)
        mal_ratio = random.uniform(0.1, 0.25)
        eps       = 1
        topology  = None
    else:
        try:
            nodes     = int(data.get('nodes', 20))
            mal_ratio = float(data.get('malicious_ratio', 0.2))
        except (ValueError, TypeError):
            return jsonify({"error": "nodes must be int, malicious_ratio must be float"}), 400
        if not (5 <= nodes <= 300):
            return jsonify({"error": "nodes must be between 5 and 300"}), 400
        if not (0.0 <= mal_ratio <= 0.9):
            return jsonify({"error": "malicious_ratio must be between 0.0 and 0.9"}), 400
        try:
            eps = int(data.get('test_episodes', 100))
            eps = max(1, min(eps, 500))
        except (ValueError, TypeError):
            eps = 100

        if nodes > 100: eps = min(eps, 5)
        elif nodes > 50: eps = min(eps, 10)
        elif nodes > 30: eps = min(eps, 20)

        topology   = _resolve_topology(data.get('topology', ''))
        multi_path = bool(data.get('multi_path', False))

    custom_env = NetworkEnvironment(
        num_nodes=nodes, malicious_ratio=mal_ratio,
        randomize_graph=True, forced_topology=topology
    )

    success, crashes = 0, 0
    demo_path = []; demo_info = {}; demo_src = demo_dst = None
    nodes_data = []; edges_data = []; alt_paths = []

    for i in range(eps):
        valid_actions, features = custom_env.reset(episode=900000 + i, is_test=True)
        done = False
        path = [custom_env.source]
        info = {"reason": "step"}

        while not done:
            action, _ = agent.act(valid_actions, features)
            if action is None:
                break
            (valid_actions, features), _, done, info = custom_env.step(action)
            path.append(custom_env.current_node)

        if info["reason"] == "success":   success += 1
        elif info["reason"] == "malicious": crashes += 1

        if i == 0:
            demo_src, demo_dst = custom_env.source, custom_env.destination
            demo_path, demo_info = path, info
            nodes_data, edges_data = _build_nodes_edges(custom_env, demo_src, demo_dst)

            if mode != 'user' and multi_path:
                valid_pairs = [p for p in custom_env._get_valid_pairs()
                               if p != (demo_src, demo_dst)]
                for a_src, a_dst in random.sample(valid_pairs, min(2, len(valid_pairs))):
                    ap, ar = _run_single_path(custom_env, a_src, a_dst)
                    alt_paths.append({"path": ap, "reason": ar, "src": a_src, "dst": a_dst})

    accuracy   = (success / eps) * 100
    crash_rate = (crashes / eps) * 100

    log.info("Simulate: nodes=%d mal=%.1f%% eps=%d acc=%.1f%% crash=%.1f%%",
             nodes, mal_ratio * 100, eps, accuracy, crash_rate)

    return jsonify({
        "accuracy":   round(accuracy, 1),
        "crash_rate": round(crash_rate, 1),
        "demo_src":   demo_src,
        "demo_dst":   demo_dst,
        "demo_path":  demo_path,
        "hop_count":  len(demo_path) - 1,
        "reason":     demo_info.get("reason", "").upper(),
        "nodes":      nodes_data,
        "edges":      edges_data,
        "alt_paths":  alt_paths,
        "node_count": len(nodes_data),
        "edge_count": len(edges_data),
        "malicious_count": sum(1 for n in nodes_data if n["group"] == "malicious"),
    })


# ─── Live routing via SSE ─────────────────────────────────────────────────────

@app.route('/stream_route')
def stream_route():
    """
    Server-Sent Events endpoint — streams individual routing hops in real-time.
    Query params:
      nodes      (int,   default 20)    — number of network nodes
      mal_ratio  (float, default 0.2)   — malicious node ratio
      topology   (str,   default random) — graph topology key
      speed_ms   (int,   default 750)   — milliseconds between hops
    """
    nodes_n   = max(5, min(request.args.get('nodes', 20, type=int), 300))
    mal_ratio = max(0.0, min(request.args.get('mal_ratio', 0.2, type=float), 0.9))
    topology  = _resolve_topology(request.args.get('topology', ''))
    speed_ms  = max(100, min(request.args.get('speed_ms', 750, type=int), 3000))

    def generate():
        stream_env = NetworkEnvironment(
            num_nodes=nodes_n, malicious_ratio=mal_ratio,
            randomize_graph=True, forced_topology=topology
        )
        valid_actions, features = stream_env.reset(is_test=True)
        src = stream_env.source
        dst = stream_env.destination

        nd, ed = _build_nodes_edges(stream_env, src, dst)
        yield f"data: {json.dumps({'type':'init','nodes':nd,'edges':ed,'source':src,'dest':dst,'node_count':len(nd),'edge_count':len(ed),'malicious_count':sum(1 for n in nd if n['group']=='malicious')})}\n\n"

        done = False
        path = [src]
        info = {"reason": "step"}

        while not done:
            action, _ = agent.act(valid_actions, features)
            if action is None:
                break
            prev = stream_env.current_node
            (valid_actions, features), _, done, info = stream_env.step(action)
            cur = stream_env.current_node
            path.append(cur)
            yield f"data: {json.dumps({'type':'step','from':prev,'to':cur,'reason':info['reason']})}\n\n"
            time.sleep(speed_ms / 1000.0)

        yield f"data: {json.dumps({'type':'done','reason':info['reason'].upper(),'path':path,'hop_count':len(path)-1})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log.info("Starting %s v%s on http://localhost:%d", APP_NAME, APP_VERSION, PORT)
    app.run(host=HOST, port=PORT, threaded=True)
