"""Flask backend for WPARK car park simulation."""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
import os
import json
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.car_park import CarPark, BayType, BayStatus
from src.models.vehicle import Vehicle, VehicleSize, VisitPurpose, MobilityLevel
from src.simulation import SimulationEngine, ASSIGNMENT_STRATEGIES, get_nn_strategy, NeuralNetworkAssignmentStrategy
from train_worker import evaluate_weights as _evaluate_weights, REFERENCE_SCORE as _REFERENCE_SCORE

app = Flask(__name__)
CORS(app)

# Global simulation state
simulation_state = {
    'engine': None,
    'animation_queue': [],  # Cars entering/exiting for animation
}


def init_simulation():
    """Initialize or reset simulation."""
    simulation_state['engine'] = SimulationEngine()
    simulation_state['animation_queue'] = []
    return simulation_state['engine']


def get_engine():
    """Get or create simulation engine."""
    if simulation_state['engine'] is None:
        init_simulation()
    return simulation_state['engine']


def vehicle_to_dict(vehicle: Vehicle) -> dict:
    """Convert vehicle to JSON-serializable dict."""
    return {
        'id': vehicle.id,
        'size': vehicle.size.value,
        'is_ev': vehicle.is_ev,
        'purpose': vehicle.purpose.value,
        'mobility': vehicle.mobility.value,
        'target_shop': vehicle.target_shop,
        'target_floor': vehicle.target_floor,
        'arrival_time': vehicle.arrival_time,
        'estimated_stay_minutes': vehicle.estimated_stay_minutes,
        'assigned_bay_id': vehicle.assigned_bay_id,
        'assigned_level': vehicle.assigned_level,
        'priority': vehicle.priority,
        'requires_blue_badge': vehicle.requires_blue_badge,
        'requires_parent_child': vehicle.requires_parent_child,
        'requires_ev_charging': vehicle.requires_ev_charging,
        'color': get_vehicle_color(vehicle),
        'width': get_vehicle_width(vehicle),
        'length': get_vehicle_length(vehicle),
    }


def get_vehicle_color(vehicle: Vehicle) -> str:
    """Get vehicle display color based on properties."""
    if vehicle.purpose == VisitPurpose.EMERGENCY:
        return '#FF4444'
    elif vehicle.requires_blue_badge:
        return '#4488FF'
    elif vehicle.requires_parent_child:
        return '#AA66CC'
    elif vehicle.is_ev:
        return '#44FF44'
    else:
        # Vary color by purpose
        colors = {
            'shopping': '#FF8800',
            'dining': '#FFAA00',
            'entertainment': '#FF6600',
            'medical': '#66AAFF',
            'quick_errand': '#FFCC00',
            'commute': '#888888'
        }
        return colors.get(vehicle.purpose.value, '#FF8800')


def get_vehicle_width(vehicle: Vehicle) -> int:
    """Get vehicle width in pixels based on size."""
    sizes = {
        VehicleSize.COMPACT: 20,
        VehicleSize.STANDARD: 24,
        VehicleSize.LARGE: 28,
        VehicleSize.OVERSIZED: 32
    }
    return sizes.get(vehicle.size, 24)


def get_vehicle_length(vehicle: Vehicle) -> int:
    """Get vehicle length in pixels based on size."""
    sizes = {
        VehicleSize.COMPACT: 32,
        VehicleSize.STANDARD: 40,
        VehicleSize.LARGE: 48,
        VehicleSize.OVERSIZED: 56
    }
    return sizes.get(vehicle.size, 40)


def bay_to_dict(bay) -> dict:
    """Convert bay to JSON-serializable dict."""
    return {
        'id': bay.id,
        'row': bay.row,
        'position': bay.position,
        'bay_type': bay.bay_type.value,
        'status': bay.status.value,
        'size': bay.size.value,
        'x': bay.x,
        'y': bay.y,
        'distance_to_lift': bay.distance_to_lift,
        'distance_to_exit': bay.distance_to_exit,
        'occupied_by': bay.occupied_by,
        'color': get_bay_color(bay),
    }


def get_bay_color(bay) -> str:
    """Get bay display color."""
    if bay.status == BayStatus.OCCUPIED:
        return '#662200'  # Dark orange for occupied
    elif bay.status == BayStatus.RESERVED:
        return '#664400'
    elif bay.status == BayStatus.MAINTENANCE:
        return '#333333'
    elif bay.bay_type == BayType.BLUE_BADGE:
        return '#003366'
    elif bay.bay_type == BayType.PARENT_CHILD:
        return '#440066'
    elif bay.bay_type == BayType.EV:
        return '#004400'
    else:
        return '#1a1a1a'  # Dark available


def level_to_dict(level, engine) -> dict:
    """Convert level to JSON-serializable dict."""
    # Get vehicles on this level
    vehicles_on_level = []
    for vid, vehicle in engine.active_vehicles.items():
        # Compare level_number (int) with assigned_level (int)
        if vehicle.assigned_level == level.level_number:
            v_dict = vehicle_to_dict(vehicle)
            # Find the bay to get position
            for bay in level.bays:
                if bay.id == vehicle.assigned_bay_id:
                    v_dict['x'] = bay.x
                    v_dict['y'] = bay.y
                    break
            vehicles_on_level.append(v_dict)
    
    # Determine if this is the main entrance level
    is_main_entrance = (level.level_number == engine.car_park.entry_level)
    
    return {
        'id': level.id,
        'level_number': level.level_number,
        'name': level.name,
        'capacity': level.capacity,
        'occupied_count': level.occupied_count,
        'available_count': level.available_count,
        'occupancy_rate': level.occupancy_rate,
        'nearby_shops': level.nearby_shops,
        'bays': [bay_to_dict(bay) for bay in level.bays],
        'vehicles': vehicles_on_level,
        # Layout info
        'width': 800,
        'height': 400,
        'entrance': {'x': level.entrance_x, 'y': level.entrance_y},
        'exit': {'x': level.exit_x, 'y': level.exit_y},
        'shop_gate': {'x': level.shop_gate_x, 'y': level.shop_gate_y},
        # Main entrance is on Level 1
        'is_main_entrance': is_main_entrance,
        'has_ramp_up': level.level_number < 1,    # Levels below 1 have ramp up
        'has_ramp_down': level.level_number > -1,  # Levels above -1 have ramp down
        'ramp_up': {'x': 50, 'y': 200},
        'ramp_down': {'x': 750, 'y': 200},
    }


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current simulation state."""
    engine = get_engine()
    
    state = engine.get_state_summary()
    
    # Add levels with bays
    state['levels'] = [level_to_dict(level, engine) for level in engine.car_park.levels]
    
    # Add active vehicles
    state['active_vehicles'] = {
        vid: vehicle_to_dict(v) for vid, v in engine.active_vehicles.items()
    }
    
    # Add waiting queue
    state['waiting_queue'] = [vehicle_to_dict(v) for v in engine.waiting_queue]
    
    # Add animation events
    state['animations'] = simulation_state['animation_queue']
    simulation_state['animation_queue'] = []  # Clear after sending

    # Add shops
    state['shops'] = [
        {
            'id': shop.id,
            'name': shop.name,
            'floor': shop.floor,
            'category': shop.category.value,
            'avg_visit_minutes': shop.avg_visit_minutes,
            'color': shop.get_display_color()
        }
        for shop in engine.shops
    ]
    
    # Stats
    state['stats'] = {
        'total_arrivals': engine.stats.total_arrivals,
        'total_departures': engine.stats.total_departures,
        'total_denied': engine.stats.total_denied,
        'avg_stay_duration': engine.stats.avg_stay_duration,
        'peak_occupancy': engine.stats.peak_occupancy,
        'arrivals_by_purpose': engine.stats.arrivals_by_purpose,
        'arrivals_by_level': engine.stats.arrivals_by_level,
        'nn_assignments': engine.stats.nn_assignments,
        'nn_avg_confidence': engine.stats.nn_avg_confidence,
        'avg_assignment_quality': engine.stats.avg_assignment_quality,
        'near_misses': engine.stats.near_misses,
    }

    # Traffic state
    state['traffic'] = {
        'entry_queue_length': len(engine.entry_queue),
        'in_transit_count':   len(engine.in_transit),
        'zone_occupancy':     dict(engine.zone_occupancy) if hasattr(engine, 'zone_occupancy') else {},
        'zone_capacity':      dict(engine._zone_capacity) if hasattr(engine, '_zone_capacity') else {},
        'avg_entry_wait':     round(engine.stats.avg_entry_wait, 1),
        'congestion_events':  engine.stats.congestion_events,
    }
    state['entry_queue'] = [vehicle_to_dict(v) for v, _ in engine.entry_queue]

    # In-transit vehicles (reserved bay but still driving to it)
    state['in_transit_vehicles'] = {
        vid: {
            'vehicle': vehicle_to_dict(info['vehicle']),
            'bay_id': info['bay_id'],
            'level': info['level'],
        }
        for vid, info in engine.in_transit.items()
    }

    return jsonify(state)


@app.route('/api/step', methods=['POST'])
def step_simulation():
    """Advance simulation by one step."""
    # Check whether the training loop just completed a generation and wants a
    # fresh simulation so the new best network is evaluated from a clean state.
    reset_happened = False
    with _training_lock:
        if training_state.get('reset_pending', False):
            training_state['reset_pending'] = False
            reset_happened = True

    if reset_happened:
        engine = init_simulation()
        engine.assignment_strategy = 'neural_network'
    else:
        engine = get_engine()

    # Support both JSON body and empty body
    if request.is_json:
        data = request.json or {}
    else:
        data = {}
    time_delta = data.get('time_delta', 1.0)
    
    # Store state before step
    old_vehicles = set(engine.active_vehicles.keys())

    # Step simulation
    engine.step(time_delta)

    new_vehicles = set(engine.active_vehicles.keys())
    departures = old_vehicles - new_vehicles

    # Enter animations come from engine._pending_animations
    for anim in engine._pending_animations:
        vid = anim['vehicle_id']
        # Vehicle may be in in_transit or just moved to active_vehicles
        vehicle = engine.in_transit.get(vid, {}).get('vehicle') or engine.active_vehicles.get(vid)
        if vehicle:
            simulation_state['animation_queue'].append({
                'type': 'enter',
                'vehicle': vehicle_to_dict(vehicle),
                'level': anim['level'],
                'bay_id': anim['bay_id'],
                'timestamp': anim['timestamp'],
            })
    engine._pending_animations.clear()

    # Exit animations from departures
    for vid in departures:
        for v in engine.departed_vehicles[-10:]:
            if v.id == vid:
                simulation_state['animation_queue'].append({
                    'type': 'exit',
                    'vehicle': vehicle_to_dict(v),
                    'level': v.assigned_level,
                    'bay_id': v.assigned_bay_id,
                    'timestamp': engine.current_time
                })
                break

    # Return both step result and accumulated animations
    animations_to_return = list(simulation_state['animation_queue'])
    simulation_state['animation_queue'] = []  # Clear after returning
    
    return jsonify({
        'success': True,
        'time': engine.current_time,
        'animations': animations_to_return,
        'reset': reset_happened,
    })


@app.route('/api/near_miss', methods=['POST'])
def report_near_miss():
    """Frontend reports a near-miss animation event."""
    engine = get_engine()
    engine.stats.near_misses += 1
    return jsonify({'total': engine.stats.near_misses})


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation."""
    init_simulation()
    return jsonify({'success': True})


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update simulation configuration."""
    engine = get_engine()
    data = request.json or {}
    
    if 'arrival_rate' in data:
        engine.arrival_rate = float(data['arrival_rate'])
    if 'assignment_strategy' in data:
        if data['assignment_strategy'] in ASSIGNMENT_STRATEGIES:
            engine.assignment_strategy = data['assignment_strategy']
    
    return jsonify({'success': True})


@app.route('/api/strategies')
def get_strategies():
    """Get available assignment strategies."""
    return jsonify(list(ASSIGNMENT_STRATEGIES.keys()))


@app.route('/api/neural_network')
def get_nn_state():
    """Get neural network state for visualization."""
    nn_strategy = get_nn_strategy()
    network = nn_strategy.network
    
    layers = []
    for layer in network.get_layer_info():
        # Get weight statistics for visualization
        weight_stats = {}
        if layer.weights is not None and layer.weights.size > 0:
            weights_flat = layer.weights.flatten()
            weight_stats = {
                'min': float(weights_flat.min()),
                'max': float(weights_flat.max()),
                'mean': float(weights_flat.mean()),
                'std': float(weights_flat.std()),
                'shape': list(layer.weights.shape),
                # Sample weights for visualization (max 100)
                'sample': weights_flat[:min(100, len(weights_flat))].tolist()
            }
        
        layers.append({
            'name': layer.name,
            'input_size': layer.input_size,
            'output_size': layer.output_size,
            'activations': layer.activations.tolist() if layer.activations is not None else [],
            'weights': weight_stats,
            'biases': layer.biases.tolist() if layer.biases is not None and layer.biases.size > 0 else []
        })
    
    prediction_info = {}
    if hasattr(nn_strategy, '_nn_assignment') and nn_strategy._nn_assignment:
        prediction_info = nn_strategy._nn_assignment.last_prediction_info
    
    return jsonify({
        'layers': layers,
        'total_params': len(network.get_weights_flat()),
        'prediction_info': prediction_info
    })


# Training state — written by background thread, read by request handlers
training_state = {
    'active': False,
    'fast_mode': False,
    'trainer': None,
    'generation': 0,
    'best_fitness': 0.0,
    'fitness_history': [],
    'population_stats': {},
    'current_individual': 0,   # how many individuals have finished this generation
    'population_size': 0,
    'parallel_workers': 0,     # how many are running right now
    'generation_reports': [],  # per-generation per-network score reports
    'reset_pending': False,    # signals /api/step to reset the live sim for the new generation
    '_stop_event': None,
    '_thread': None,
}
_training_lock = threading.Lock()

# _MAX_WORKERS: use all available cores; capped at population size at runtime
_MAX_WORKERS = os.cpu_count() or 4


def _run_training_loop(stop_event, fast_mode: bool = False):
    """
    Background training thread.
    Each generation:
      1. Evaluate all individuals in parallel (realistic day sim each)
      2. Build per-network score report
      3. Evolve population (elitism + crossover + mutation)
      4. Push best network to live simulation
    """
    # Use all available CPU cores — ProcessPoolExecutor bypasses the GIL so
    # each worker truly runs in parallel (one core each).
    n_workers = min(_MAX_WORKERS, 20)   # cap at 20 so we don't over-subscribe

    # Create the pool ONCE and reuse it across all generations to avoid the
    # per-generation process-spawn overhead (~0.5 s per worker on Windows).
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        while not stop_event.is_set():
            with _training_lock:
                trainer = training_state['trainer']
                if trainer is None:
                    break
                population = list(trainer.population)
                pop_size   = len(population)

            with _training_lock:
                training_state['population_size']    = pop_size
                training_state['current_individual'] = 0
                training_state['parallel_workers']   = min(n_workers, pop_size)

            # ── Parallel evaluation ──────────────────────────────────────
            completed = 0
            futures   = {}

            for ind in population:
                if stop_event.is_set():
                    break
                # Pass weights (numpy array — picklable) to the worker process.
                f = executor.submit(_evaluate_weights, ind.weights, fast_mode)
                futures[f] = ind

            for f in as_completed(futures):
                if stop_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
                ind = futures[f]
                try:
                    ind.fitness = f.result()
                except Exception:
                    ind.fitness = 0.0
                completed += 1
                with _training_lock:
                    training_state['current_individual'] = completed

            with _training_lock:
                training_state['parallel_workers'] = 0

            if stop_event.is_set():
                break

            # ── Build per-network generation report (before evolution) ───
            with _training_lock:
                sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
                next_gen   = training_state['generation'] + 1
                pop_fitnesses = [ind.fitness for ind in population]
                avg_fit    = sum(pop_fitnesses) / max(len(pop_fitnesses), 1)

                report = {
                    'generation':    next_gen,
                    'timestamp':     time.time(),
                    'best_fitness':  round(sorted_pop[0].fitness if sorted_pop else 0.0, 1),
                    'avg_fitness':   round(avg_fit, 1),
                    'individuals': [
                        {
                            'rank':             rank + 1,
                            'fitness':          round(ind.fitness, 1),
                            'score':            max(1, min(100, round(ind.fitness / _REFERENCE_SCORE * 100))),
                            'generation_born':  ind.generation,
                            'mutation_rate':    round(getattr(ind, 'mutation_rate_used', 0.1), 3),
                        }
                        for rank, ind in enumerate(sorted_pop)
                    ],
                }
                training_state['generation_reports'].append(report)
                if len(training_state['generation_reports']) > 30:
                    training_state['generation_reports'] = training_state['generation_reports'][-30:]

            # ── Evolve & publish stats ───────────────────────────────────
            with _training_lock:
                trainer.evolve_generation()
                trainer.update_stats()

                training_state['generation']    = trainer.generation
                training_state['best_fitness']  = float(trainer.stats.best_fitness)
                training_state['fitness_history'].append(float(trainer.stats.best_fitness))
                training_state['population_stats'] = {
                    'best':      float(trainer.stats.best_fitness),
                    'avg':       float(trainer.stats.avg_fitness),
                    'worst':     float(trainer.stats.worst_fitness),
                    'std':       float(trainer.stats.fitness_std),
                    'diversity': float(trainer.get_population_diversity()),
                }
                # Push best network into the live simulation and schedule a
                # fresh-sim reset so the UI shows the new generation from t=0.
                get_nn_strategy().network = trainer.get_best_network()
                training_state['reset_pending'] = True

    with _training_lock:
        training_state['active']             = False
        training_state['current_individual'] = 0
        training_state['parallel_workers']   = 0


@app.route('/api/training/generation_report')
def get_generation_report():
    """Get per-network scores for each completed generation."""
    with _training_lock:
        reports = list(training_state.get('generation_reports', []))
    return jsonify({
        'reports':            reports[-10:],
        'latest':             reports[-1] if reports else None,
        'total_generations':  len(reports),
    })


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start genetic algorithm training in a background thread."""
    from src.neural_network import GeneticTrainer, GeneticConfig

    # Stop any existing training first
    with _training_lock:
        old_event = training_state.get('_stop_event')
    if old_event:
        old_event.set()

    data = request.json or {}
    config = GeneticConfig(
        population_size=data.get('population_size', 30),
        elite_count=data.get('elite_count', 4),
        mutation_rate=data.get('mutation_rate', 0.15),
        crossover_rate=data.get('crossover_rate', 0.7)
    )
    trainer = GeneticTrainer(config)
    trainer.initialize_population()

    fast_mode  = bool(data.get('fast_mode', False))
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_run_training_loop, args=(stop_event, fast_mode), daemon=True
    )

    with _training_lock:
        training_state['trainer']            = trainer
        training_state['active']             = True
        training_state['fast_mode']          = fast_mode
        training_state['generation']         = 0
        training_state['best_fitness']       = 0.0
        training_state['fitness_history']    = []
        training_state['population_stats']   = {}
        training_state['current_individual'] = 0
        training_state['population_size']    = config.population_size
        training_state['generation_reports'] = []
        training_state['reset_pending']      = False
        training_state['_stop_event']        = stop_event
        training_state['_thread']            = thread

    thread.start()
    return jsonify({'success': True, 'message': 'Training started',
                    'fast_mode': fast_mode})


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Signal the background training thread to stop."""
    with _training_lock:
        stop_event = training_state.get('_stop_event')
        training_state['active'] = False

    if stop_event:
        stop_event.set()

    return jsonify({'success': True})


@app.route('/api/training/status')
def training_status():
    """Get training status (fast, safe to poll frequently)."""
    with _training_lock:
        return jsonify({
            'active':             training_state['active'],
            'fast_mode':          training_state['fast_mode'],
            'generation':         training_state['generation'],
            'best_fitness':       training_state['best_fitness'],
            'current_individual': training_state['current_individual'],
            'population_size':    training_state['population_size'],
            'parallel_workers':   training_state['parallel_workers'],
            'fitness_history':    training_state['fitness_history'][-50:],
            'stats':              training_state['population_stats'],
        })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
