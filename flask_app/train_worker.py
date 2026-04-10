"""
Worker module for parallel training evaluation.

Kept separate from app.py so ProcessPoolExecutor can import it in fresh
worker processes without triggering Flask app initialisation.
"""

import os
import sys
import random

# ── Ensure the project root is on sys.path ─────────────────────────────────
# This file lives in flask_app/, so project root is one level up.
_FLASK_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_FLASK_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.simulation import SimulationEngine, NeuralNetworkAssignmentStrategy
from src.neural_network import NeuralNetwork

# ── Realistic daily traffic profile ────────────────────────────────────────
_DAILY_PROFILE = [
    ( 0,  6,  0.2),
    ( 6,  8,  1.2),
    ( 8, 10,  5.5),
    (10, 12,  2.8),
    (12, 14,  4.2),
    (14, 17,  2.2),
    (17, 19,  5.8),
    (19, 21,  2.8),
    (21, 24,  0.6),
]

_SCENARIOS = [
    {'name': 'Regular Weekday',  'multiplier': 1.0,  'weight': 4},
    {'name': 'Busy Saturday',    'multiplier': 1.65, 'weight': 2},
    {'name': 'Quiet Monday',     'multiplier': 0.50, 'weight': 1},
    {'name': 'Event Day',        'multiplier': 2.10, 'weight': 1},
    {'name': 'Bank Holiday',     'multiplier': 1.35, 'weight': 1},
    {'name': 'Early Close Day',  'multiplier': 0.80, 'weight': 1},
]

REFERENCE_SCORE = 5000.0

# Number of full days to simulate per evaluation (normal / fast mode).
# More days → better signal, slower evaluation.
_N_EVAL_DAYS      = 3   # non-fast: 3 × 24 h at 5-min steps (864 steps total)
_N_EVAL_DAYS_FAST = 2   # fast: 2 × 3-peak-window passes


def _get_arrival_rate(sim_minute: float, multiplier: float = 1.0) -> float:
    hour = (sim_minute % 1440) / 60.0
    for start_h, end_h, rate in _DAILY_PROFILE:
        if start_h <= hour < end_h:
            return max(0.05, rate * multiplier)
    return 0.2 * multiplier


def _score_engine(engine) -> float:
    """
    Compute fitness score from a finished SimulationEngine.

    The dominant term is avg_assignment_quality — a per-assignment 0-1 score
    that measures how well the NN matched each vehicle to a bay (floor proximity,
    bay type match, shop-gate proximity weighted by visit purpose).  This is the
    only term that genuinely varies with NN decision quality; all other terms
    that just measure 'did the car park fill up' have been removed.

    Approximate ranges for reference:
      random NN:  avg_assignment_quality ≈ 0.68  →  score ≈ 3 200 / 5 000 ≈ 64/100
      good NN:    avg_assignment_quality ≈ 0.85  →  score ≈ 4 300 / 5 000 ≈ 86/100
      optimal NN: avg_assignment_quality ≈ 0.95  →  score ≈ 4 900 / 5 000 ≈ 98/100
    """
    stats     = engine.stats
    arrivals  = max(stats.total_arrivals, 1)
    service_r = 1.0 - (stats.total_denied / arrivals)

    return max(
        stats.avg_assignment_quality               * 4000.0   # 0–4000 — main NN signal
        + service_r                                *  500.0   # 0–500  — reward low denial rate
        + stats.nn_avg_confidence                  *  200.0   # 0–200  — reward confident picks
        + max(0.0, 1.0 - stats.avg_entry_wait / 15.0) * 300.0,  # 0–300 — penalise queuing
        0.0,
    )


def _make_engine(weights):
    """Build a SimulationEngine wired to the given flat weight array."""
    import numpy as np
    engine   = SimulationEngine()
    engine.assignment_strategy = 'neural_network'
    nn_strat = NeuralNetworkAssignmentStrategy()
    network  = NeuralNetwork()
    network.set_weights_flat(np.asarray(weights))
    nn_strat.network = network
    if nn_strat._nn_assignment is not None:
        nn_strat._nn_assignment.training_mode = True
    engine._nn_strategy_override = nn_strat
    return engine


def evaluate_weights(weights, fast_mode: bool = False) -> float:
    """
    Evaluate one neural network (flat weight array) over multiple simulated days.

    This is a top-level, importable function so ProcessPoolExecutor can
    pickle and send it to worker processes.

    fast_mode=False: 5-min steps, 3 independent 24-hour days  (3 × 288 = 864 steps)
    fast_mode=True:  10-min steps, 2 × 3-peak-window passes   (2 ×  54 = 108 steps)

    Scores are averaged across days so the REFERENCE_SCORE scale is unchanged.
    """
    n_days   = _N_EVAL_DAYS_FAST if fast_mode else _N_EVAL_DAYS
    step_min = 10.0 if fast_mode else 5.0
    scores   = []

    for _ in range(n_days):
        scenario = random.choices(_SCENARIOS, weights=[s['weight'] for s in _SCENARIOS])[0]
        mult     = scenario['multiplier']
        engine   = _make_engine(weights)

        if fast_mode:
            for win_start, win_end in ((7*60, 10*60), (12*60, 14*60), (17*60, 20*60)):
                t = win_start
                while t < win_end:
                    engine.arrival_rate = _get_arrival_rate(t, mult)
                    engine.step(step_min)
                    t += step_min
        else:
            sim_minute = 0.0
            for _ in range(int(24 * 60 / step_min)):
                engine.arrival_rate = _get_arrival_rate(sim_minute, mult)
                engine.step(step_min)
                sim_minute += step_min

        scores.append(_score_engine(engine))

    return sum(scores) / len(scores)
