"""
config.py – All hyper-parameters and path constants for reward_metrics_svc.
Every numeric parameter is overridable via environment variable.
"""

import logging
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ICD10_TREE_PATH = os.path.join(os.path.dirname(__file__), "icd10_tree.json")

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE DISTANCE WEIGHTS  (spec §11, must sum > 0)
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = float(os.getenv("ALPHA", "0.4"))  # D_set weight
BETA = float(os.getenv("BETA", "0.3"))  # coverage penalty weight
GAMMA = float(os.getenv("GAMMA", "0.2"))  # extra-codes penalty weight
DELTA = float(os.getenv("DELTA", "0.1"))  # cardinality penalty weight
WEIGHT_SUM = ALPHA + BETA + GAMMA + DELTA  # normalisation denominator

# ─────────────────────────────────────────────────────────────────────────────
# PENALTY LAMBDAS  (spec §9, §10)
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_EXTRA = float(os.getenv("LAMBDA_EXTRA", "0.4"))  # ∈ [0.3, 0.5]
LAMBDA_CARD = float(os.getenv("LAMBDA_CARD", "0.3"))  # ∈ [0.2, 0.4]

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID REWARD WEIGHTS  (spec §13)
# ─────────────────────────────────────────────────────────────────────────────
W_TREE = float(os.getenv("W_TREE", "0.6"))  # tree-distance improvement
W_EXACT = float(os.getenv("W_EXACT", "0.25"))  # Jaccard exact-match
W_STRUCTURE = float(os.getenv("W_STRUCTURE", "0.15"))  # structural validity

# ─────────────────────────────────────────────────────────────────────────────
# ROLLOUT QUEUE  (spec §16)
# ─────────────────────────────────────────────────────────────────────────────
RL_SERVICE_URL = os.getenv("RL_SERVICE_URL", "http://localhost:8003/ingest_rollout")
ROLLOUT_DIR = os.path.join(os.path.dirname(__file__), "rollouts")

# ─────────────────────────────────────────────────────────────────────────────
# TREE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VIRTUAL_ROOT = "__ROOT__"  # synthetic ancestor linking all top-level ICD codes

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING FORMAT  (applied lazily – only the first basicConfig call wins)
# ─────────────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure the root logger.  No-op if handlers are already attached."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
