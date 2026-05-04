"""
Central configuration for the Neural Router project.
All values can be overridden via environment variables.
Imported by app.py, train.py, and neural_router.py.
"""

import os

# ─── App Identity ──────────────────────────────────────────────────────────────
APP_NAME    = "Neural Router"
APP_VERSION = "2.0.0"

# ─── Network Topology ─────────────────────────────────────────────────────────
NUM_NODES       = int(os.environ.get("NUM_NODES",       15))
MALICIOUS_RATIO = float(os.environ.get("MALICIOUS_RATIO", 0.2))

# ─── Agent ────────────────────────────────────────────────────────────────────
FEATURE_SIZE   = int(os.environ.get("FEATURE_SIZE",   5))   # [trust, is_dest, degree, dist_norm, steps_norm]
TRAIN_EPISODES = int(os.environ.get("TRAIN_EPISODES", 1500))
GAMMA          = float(os.environ.get("GAMMA",          0.95))
LR             = float(os.environ.get("LR",             0.001))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE",     64))
EPSILON_START  = float(os.environ.get("EPSILON_START",  1.0))
EPSILON_MIN    = float(os.environ.get("EPSILON_MIN",    0.01))
EPSILON_DECAY  = float(os.environ.get("EPSILON_DECAY",  0.995))
TAU            = float(os.environ.get("TAU",            0.01))  # Soft target update coefficient

# ─── PER (Prioritized Experience Replay) ─────────────────────────────────────
PER_ALPHA      = float(os.environ.get("PER_ALPHA",      0.6))   # Priority exponent (0 = uniform, 1 = fully prioritized)
PER_BETA_START = float(os.environ.get("PER_BETA_START", 0.4))   # IS weight exponent (annealed to 1.0)
PER_BETA_STEPS = int(os.environ.get("PER_BETA_STEPS", 50000))   # Steps over which beta → 1.0
REPLAY_CAPACITY = int(os.environ.get("REPLAY_CAPACITY", 50000))

# ─── Files ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "router_model.pth")

# ─── Server ───────────────────────────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5001))
