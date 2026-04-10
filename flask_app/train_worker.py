"""
Worker module for parallel training evaluation.

Kept separate from app.py so ProcessPoolExecutor can import it in fresh
worker processes without triggering Flask app initialisation.
"""

import os
import sys
import random
import numpy as np

# ── Ensure the project root is on sys.path ─────────────────────────────────
# This file lives in flask_app/, so project root is one level up.
_FLASK_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_FLASK_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.simulation import SimulationEngine, NeuralNetworkAssignmentStrategy
from src.neural_network import NeuralNetwork

# ── Realistic daily traffic profile ────────────────────────────────────────
# Rates are vehicles/minute.  Car park capacity = 212 bays; sustainable peak
# (80 % occupancy, ~90 min avg stay) ≈ 1.9 /min.  Peaks are set just above
# that so the car park runs 85-95 % full during rush hours — busy enough to
# create meaningful assignment decisions, but not so overwhelmed that the NN's
# choices become irrelevant and service_r collapses.
_DAILY_PROFILE = [
    ( 0,  6,  0.08),
    ( 6,  8,  0.5),
    ( 8, 10,  2.2),
    (10, 12,  1.1),
    (12, 14,  1.7),
    (14, 17,  0.9),
    (17, 19,  2.4),
    (19, 21,  1.1),
    (21, 24,  0.2),
]

_SCENARIOS = [
    {'name': 'Regular Weekday',  'multiplier': 1.0,  'weight': 4},
    {'name': 'Busy Saturday',    'multiplier': 1.3,  'weight': 2},  # was 1.65 — too overwhelming
    {'name': 'Quiet Monday',     'multiplier': 0.55, 'weight': 1},
    {'name': 'Event Day',        'multiplier': 1.5,  'weight': 1},  # was 2.10
    {'name': 'Bank Holiday',     'multiplier': 1.2,  'weight': 1},
    {'name': 'Early Close Day',  'multiplier': 0.75, 'weight': 1},
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

    Terms:
      avg_assignment_quality  0–4500  Main NN signal: floor proximity + bay-type match + lift proximity
      service_r               0–500   Fraction of total demand that was served (always 0–1)

    nn_avg_confidence was removed: it rewarded peaked softmax outputs regardless of
    decision quality, giving the NN a free +160 pts that deterministic strategies
    don't receive.  The 200 pts were redistributed to avg_assignment_quality.

    Realistic ranges (random NN baseline ~50–55, smart ceiling ~58–62):
      random NN:  avg_assignment_quality ≈ 0.75  →  score ≈ 3 375 / 5 000 ≈ 67/100
      good NN:    avg_assignment_quality ≈ 0.88  →  score ≈ 4 460 / 5 000 ≈ 89/100
    """
    stats        = engine.stats
    total_demand = stats.total_arrivals + stats.total_denied
    # service_r: fraction of all demand that was served (always 0–1)
    service_r    = stats.total_arrivals / max(total_demand, 1)

    return max(
        stats.avg_assignment_quality * 4500.0   # 0–4500 — sole NN signal
        + service_r                *  500.0,    # 0–500  — reward low denial rate
        0.0,
    )


def _make_engine(weights):
    """Build a SimulationEngine wired to the given flat weight array."""
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

    Each call re-seeds Python's (and NumPy's) RNG from OS entropy so that
    every NN in a generation faces a genuinely independent simulation — worker
    processes are reused across generations, which would otherwise carry
    correlated random state from one evaluation to the next.

    fast_mode=False: 5-min steps, 3 independent 24-hour days  (3 × 288 = 864 steps)
    fast_mode=True:  10-min steps, 2 × 3-peak-window passes   (2 ×  54 = 108 steps)

    Scores are averaged across days so the REFERENCE_SCORE scale is unchanged.
    """
    # Fresh, OS-seeded randomness for every evaluation — guaranteed independence
    # regardless of which worker process handles this call.
    random.seed()
    np.random.seed()

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
