"""Project-wide constants. Everything tunable lives here so the report can cite one file."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("LIFELINE_DATA", ROOT / "data"))
ARTIFACTS = DATA_DIR / "artifacts"

# Data splits: train 2022-2024, calibrate/validate 2025, held-out test Jan-Sep 2026.
TRAIN_START = "2022-01-01"
VAL_START = "2025-01-01"
TEST_START = "2026-01-01"
TEST_END = "2026-10-01"

# Forecast: 2-hour horizon in hourly slots.
HORIZON = 2
N_SCENARIOS = 50

# Operations
RESPONSE_TARGET_S = 480          # 8-minute target for high-acuity calls
EPOCH_MIN = 15                   # decision epoch
FLEET_SIZE = int(os.getenv("LIFELINE_FLEET", "0"))  # 0 = use calibrated value
MAX_MOVES = 3                    # macro-move size per decision
COVERAGE_FLOOR = 0.5             # each zone must have >= this P(covered within target)
MAX_CRITIC_ROUNDS = 3

# MCTS
MCTS_ITERS = int(os.getenv("LIFELINE_MCTS_ITERS", "600"))
MCTS_C = 1.2
CANDIDATE_POSTS = 15

# Reward weights: -mean_rt + L1*within_target - L2*gini - L3*km_moved
L1_WITHIN = 120.0
L2_GINI = 400.0
L3_KM = float(os.getenv("LIFELINE_L3_KM", "1.0"))   # crew-burden cost per km moved (tuned in eval/tune.py)

# Autonomy tiers
TIER1_MAX_UNITS, TIER1_MAX_KM = 1, 2.0
TIER2_MAX_UNITS = 3
CONFIDENCE_AUTO = 0.7
