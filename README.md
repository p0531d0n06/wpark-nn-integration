# WPARK Neural Network Integration

WPARK is a **Flask + JavaScript** car park simulation that models arrivals, parking-bay assignment, congestion, and departures across a 3-level car park.  
It includes multiple assignment strategies (`random`, `greedy`, `smart`, `neural_network`) and a built-in genetic training loop for evolving the neural network in the background.

## What the current codebase provides

- Interactive web simulation at `http://127.0.0.1:5000`
- 3 levels (`-1`, `0`, `1`) with a total of **212 bays**
- Bay types: standard, blue-badge, parent/child, EV
- Vehicle generation with purpose, mobility, size, EV status, target shop/floor, and stay duration
- Traffic modeling with entry queue, in-transit vehicles, ramp/gate capacity, and congestion counters
- Neural network visualization panel (layers, activations, confidence, fitness history)
- Genetic algorithm training with parallel evaluation using `ProcessPoolExecutor`

## Architecture at a glance

- `flask_app/app.py`: Flask server, API routes, simulation lifecycle, training thread control
- `flask_app/train_worker.py`: worker-side NN fitness evaluation for parallel processing
- `flask_app/templates/index.html`: main UI shell
- `flask_app/static/js/app.js`: simulation UI orchestration, polling, training controls
- `flask_app/static/js/carParkRenderer.js`: SVG rendering + vehicle animation logic
- `src/models/*`: core data models (`Vehicle`, `Shop`, `CarPark`, `ParkingBay`)
- `src/simulation/__init__.py`: assignment strategies + `SimulationEngine`
- `src/neural_network/network.py`: NN model, encoding, inference, save/load helpers
- `src/neural_network/genetic.py`: genetic trainer, mutation/diversity/stagnation logic

## Quick start

### 1) Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Run the app

Option A:

```powershell
python flask_app\app.py
```

Option B (Windows helper script):

```powershell
.\run_flask.bat
```

Then open:

`http://127.0.0.1:5000`

## Simulation and training behavior

- Default arrival rate in engine: `2.0` vehicles/min
- Default strategy in engine: `smart`
- Training endpoint runs continuously until stopped
- Frontend "Train" starts with:
  - `population_size = 30`
  - `elite_count = 4`
  - optional `fast_mode` (from the ⚡ Fast toggle)
- Training evaluations are executed in parallel worker processes and best NN is pushed back into live simulation

## API endpoints used by the frontend

- `GET /api/state` – full current simulation state
- `POST /api/step` – advance simulation by `time_delta`
- `POST /api/reset` – reset simulation
- `POST /api/config` – update `arrival_rate` / `assignment_strategy`
- `GET /api/strategies` – list available assignment strategies
- `GET /api/neural_network` – NN layer stats, params, prediction info
- `POST /api/near_miss` – increment near-miss counter
- `POST /api/training/start` – start GA training
- `POST /api/training/stop` – stop GA training
- `GET /api/training/status` – poll training progress/stats
- `GET /api/training/generation_report` – per-generation network score reports

## Notes

- `config/default_config.json` exists but is **not currently loaded** by the runtime path in `flask_app/app.py` / `src/simulation`.
- `requirements.txt` includes `streamlit`, but this repository currently runs through Flask UI files in `flask_app/`.

## License

See `LICENSE`.
