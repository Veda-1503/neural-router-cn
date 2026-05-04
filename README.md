# Neural Router

> **AI-powered network packet routing using Deep Reinforcement Learning** — a Double DQN agent that learns to navigate dynamic network topologies while avoiding malicious nodes.

![Training Curves](https://raw.githubusercontent.com/Veda-1503/neural-router-cn/main/training_curves.png)

---

## Overview

Neural Router is a Computer Networks course project that applies **Deep Q-Learning (DQN)** to the problem of intelligent packet routing. Given a randomly generated network graph with some fraction of malicious/compromised nodes, the agent learns to find safe, efficient paths from source to destination.

**Key features:**
- **Double DQN** with Prioritized Experience Replay (PER) and soft target updates
- **Three graph topologies** — Barabási–Albert, Erdős–Rényi, Watts–Strogatz
- **Live routing via SSE** — watch the agent route hop-by-hop in real time
- **Admin dashboard** — retrain the model in the background without restarting
- **Docker-ready** — single command to build and run

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Flask Backend                     │
│                                                      │
│  app.py          ← REST API + SSE streaming          │
│  neural_router.py ← DQN Agent + PER Buffer           │
│  network_env.py  ← NetworkX Graph Environment        │
│  config.py       ← Hyperparameters (env-overridable) │
│  train.py        ← Standalone training script        │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │  Frontend (UI) │
       │  index.html    │  ← User simulation dashboard
       │  admin.html    │  ← Admin / retrain panel
       │  user.js       │  ← Vis.js network graph
       │  admin.js      │  ← Retrain controls & progress
       │  style.css     │  ← Styling
       └────────────────┘
```

### RL Components

| Component | Detail |
|---|---|
| **State** | Per-neighbour feature vector: `[trust, is_dest, degree, dist_norm, steps_norm]` |
| **Action** | Select next hop from unvisited neighbours |
| **Reward** | `+10` (reach dest) · `+speed_bonus` · `-50` (malicious) · `-10` (timeout) · `-1+trust_bonus` (step) |
| **Network** | 3-layer MLP → single Q-value per candidate |
| **Algorithm** | Double DQN + PER (SumTree) + Huber loss + soft target update |

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Veda-1503/neural-router-cn.git
cd neural-router-cn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (optional — app auto-trains on first boot)
python train.py --episodes 1500

# 4. Start the server
python app.py
```

Open **http://localhost:5001** in your browser.

### Docker

```bash
# Build
docker build -t neural-router .

# Run
docker run -p 5000:5000 neural-router
```

Open **http://localhost:5000**.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/status` | App info, model status, uptime |
| `POST` | `/simulate` | Run batch simulation, get stats + graph |
| `GET` | `/stream_route` | SSE stream — live hop-by-hop routing |
| `POST` | `/retrain` | Kick off background model retraining |
| `GET` | `/retrain/status` | Poll retrain progress |
| `GET` | `/` | User dashboard |
| `GET` | `/admin` | Admin panel |

### `/simulate` Request Body

```json
{
  "nodes": 20,
  "malicious_ratio": 0.2,
  "test_episodes": 100,
  "topology": "barabasi_albert",
  "multi_path": false
}
```

### `/stream_route` Query Params

| Param | Default | Range | Description |
|---|---|---|---|
| `nodes` | 20 | 5–300 | Number of nodes |
| `mal_ratio` | 0.2 | 0.0–0.9 | Fraction of malicious nodes |
| `topology` | random | ba / er / ws | Graph topology |
| `speed_ms` | 750 | 100–3000 | Delay between hops (ms) |

---

## Configuration

All hyperparameters live in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `NUM_NODES` | 15 | Default graph size |
| `MALICIOUS_RATIO` | 0.2 | Fraction of malicious nodes |
| `TRAIN_EPISODES` | 1500 | Training episodes |
| `GAMMA` | 0.95 | Discount factor |
| `LR` | 0.001 | Adam learning rate |
| `BATCH_SIZE` | 64 | Replay batch size |
| `EPSILON_DECAY` | 0.995 | Exploration decay rate |
| `TAU` | 0.01 | Soft target update coefficient |
| `PER_ALPHA` | 0.6 | PER priority exponent |
| `REPLAY_CAPACITY` | 50000 | Replay buffer size |
| `PORT` | 5001 | Server port |

---

## Project Structure

```
neural-router-cn/
├── app.py              # Flask application & API routes
├── neural_router.py    # DQN agent, SumTree, PER buffer, Q-network
├── network_env.py      # NetworkX-based routing environment
├── config.py           # Centralized hyperparameter config
├── train.py            # Standalone training + plotting script
├── router_model.pth    # Pre-trained model weights
├── training_curves.png # Reward / success-rate / epsilon plots
├── index.html          # User simulation dashboard
├── admin.html          # Admin / retrain panel
├── user.js             # Frontend graph & simulation logic
├── admin.js            # Admin panel JS
├── style.css           # Shared styles
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
└── test_api.py         # API test suite
```

---

## Testing

```bash
pytest test_api.py -v
```

---

## Training Results

The agent is trained for 1500 episodes across randomly generated graphs. The `training_curves.png` shows:
- **Episode reward** (raw + 50-episode rolling average)
- **Routing success rate** (target: 90%+)
- **Epsilon (ε) decay** — exploration → exploitation

---

## Tech Stack

- **Backend:** Python, Flask, PyTorch, NetworkX, NumPy
- **Frontend:** HTML/CSS/JS, Vis.js
- **ML:** Double DQN, Prioritized Experience Replay, Huber Loss
- **Infra:** Docker, Gunicorn

---

## License

This project is licensed under the [MIT License](LICENSE).
