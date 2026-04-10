"""Simulation engine for car park operations."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from functools import lru_cache
import random
import numpy as np

from ..models.car_park import CarPark, Level, ParkingBay, BayType, BayStatus
from ..models.vehicle import Vehicle, VehicleSize, VisitPurpose, MobilityLevel
from ..models.shop import Shop, generate_random_shops

if TYPE_CHECKING:
    from ..neural_network import NeuralNetwork


@dataclass
class SimulationStats:
    """Statistics for the simulation."""
    total_arrivals: int = 0
    total_departures: int = 0
    total_denied: int = 0  # Couldn't find parking
    
    avg_wait_time: float = 0.0
    avg_search_time: float = 0.0
    avg_stay_duration: float = 0.0
    
    peak_occupancy: float = 0.0
    
    # By purpose
    arrivals_by_purpose: Dict[str, int] = field(default_factory=dict)
    
    # By level
    arrivals_by_level: Dict[int, int] = field(default_factory=dict)
    
    # Neural network stats
    nn_assignments: int = 0
    nn_avg_confidence: float = 0.0

    # Per-assignment goodness-of-fit (0-1): how well each bay matched the vehicle
    avg_assignment_quality: float = 0.0

    avg_entry_wait: float = 0.0       # average minutes spent in entry queue
    peak_entry_queue: int = 0          # max entry queue length
    congestion_events: int = 0         # times a vehicle had to queue at gate/ramp
    near_misses: int = 0               # near-collision events (set by frontend via API)


class AssignmentStrategy:
    """Base class for parking assignment strategies."""
    
    def assign(self, vehicle: Vehicle, car_park: CarPark, shops: List[Shop], 
               shop_floor_map: Dict[str, int]) -> Optional[str]:
        """Assign a bay to a vehicle. Returns bay_id or None."""
        raise NotImplementedError
    
    def _get_target_floor(self, vehicle: Vehicle, shop_floor_map: Dict[str, int]) -> int:
        """Get target floor efficiently using precomputed shop map."""
        if vehicle.target_floor is not None:
            return vehicle.target_floor
        if vehicle.target_shop and vehicle.target_shop in shop_floor_map:
            return shop_floor_map[vehicle.target_shop]
        return 0  # Default to ground


class RandomAssignment(AssignmentStrategy):
    """Randomly assign to any available bay."""
    
    def assign(self, vehicle: Vehicle, car_park: CarPark, shops: List[Shop],
               shop_floor_map: Dict[str, int]) -> Optional[str]:
        available = car_park.get_all_available_bays()
        if not available:
            return None
        return random.choice(available).id


class GreedyAssignment(AssignmentStrategy):
    """Assign closest available bay to target floor."""
    
    def assign(self, vehicle: Vehicle, car_park: CarPark, shops: List[Shop],
               shop_floor_map: Dict[str, int]) -> Optional[str]:
        # Determine required bay type
        required_type = None
        if vehicle.requires_blue_badge:
            required_type = BayType.BLUE_BADGE
        elif vehicle.requires_parent_child:
            required_type = BayType.PARENT_CHILD
        elif vehicle.requires_ev_charging:
            required_type = BayType.EV
        
        target_floor = self._get_target_floor(vehicle, shop_floor_map)
        
        # Sort levels by distance to target floor for early exit
        sorted_levels = sorted(car_park.levels, 
                               key=lambda l: abs(l.level_number - target_floor))
        
        best_bay = None
        best_score = float('inf')
        
        for level in sorted_levels:
            floor_distance = abs(level.level_number - target_floor)
            # Early exit if we found a bay on closer floor
            if best_bay and floor_distance * 50 > best_score:
                break
                
            available = level.get_available_bays()
            
            # Filter by type if required
            if required_type:
                type_bays = [b for b in available if b.bay_type == required_type]
                if type_bays:
                    available = type_bays
                elif required_type != BayType.STANDARD:
                    available = [b for b in available if b.bay_type == BayType.STANDARD]
            
            for bay in available:
                score = floor_distance * 50 + bay.distance_to_lift
                if vehicle.is_ev and bay.bay_type == BayType.EV:
                    score -= 100
                
                if score < best_score:
                    best_score = score
                    best_bay = bay
        
        return best_bay.id if best_bay else None


class SmartAssignment(AssignmentStrategy):
    """Smart assignment considering multiple factors including vehicle size and shop gate distance.
    
    Note: All vehicles enter from Level 1 (First Floor) and must travel via ramps
    to reach other floors. This adds travel time for floors other than Level 1.
    """
    
    # Main entrance is on Level 1
    ENTRY_LEVEL = 1
    
    def assign(self, vehicle: Vehicle, car_park: CarPark, shops: List[Shop],
               shop_floor_map: Dict[str, int]) -> Optional[str]:
        target_floor = self._get_target_floor(vehicle, shop_floor_map)
        
        # Sort levels by combined distance: from entry (Level 1) + to target shop
        # This accounts for having to enter via Level 1
        def level_priority(level):
            # Distance from entry level (Level 1) to this floor
            entry_distance = abs(level.level_number - self.ENTRY_LEVEL)
            # Distance from this floor to target shop floor
            shop_distance = abs(level.level_number - target_floor)
            return entry_distance + shop_distance * 1.5  # Weight shop proximity higher
        
        sorted_levels = sorted(car_park.levels, key=level_priority)
        
        best_bay = None
        best_score = float('inf')
        
        # Pre-check vehicle requirements
        needs_blue_badge = vehicle.requires_blue_badge
        needs_parent_child = vehicle.requires_parent_child
        is_ev = vehicle.is_ev
        priority_bonus = vehicle.priority * 30
        
        # Vehicle size penalty (larger vehicles prefer larger/more accessible bays)
        size_scores = {
            VehicleSize.COMPACT: 0,
            VehicleSize.STANDARD: 10,
            VehicleSize.LARGE: 30,
            VehicleSize.OVERSIZED: 60
        }
        size_penalty = size_scores.get(vehicle.size, 10)
        
        for level in sorted_levels:
            # Distance from entry level (cars enter from Level 1)
            entry_distance = abs(level.level_number - self.ENTRY_LEVEL)
            # Distance to target shop floor
            shop_distance = abs(level.level_number - target_floor)
            
            # Base score: travel from entry + travel to shop (weighted)
            base_floor_score = entry_distance * 50 + shop_distance * 100
            
            # Early exit optimization
            if best_bay and base_floor_score > best_score:
                break
            
            available = level.get_available_bays()
            if not available:
                continue
                
            occupancy_penalty = level.occupancy_rate * 50
            
            for bay in available:
                score = base_floor_score
                
                # Bay type matching (optimized conditionals)
                if needs_blue_badge:
                    score += -200 if bay.bay_type == BayType.BLUE_BADGE else 500
                elif needs_parent_child:
                    score += -150 if bay.bay_type == BayType.PARENT_CHILD else 300
                elif is_ev and bay.bay_type == BayType.EV:
                    score -= 100
                
                # Distance to shop gate (lift) - primary factor for convenience
                score += bay.distance_to_lift * 0.3
                
                # Distance to exit for quick errands
                if vehicle.purpose == VisitPurpose.QUICK_ERRAND:
                    score += bay.distance_to_exit * 0.2
                
                # Vehicle size consideration - larger vehicles prefer end positions
                if vehicle.size in (VehicleSize.LARGE, VehicleSize.OVERSIZED):
                    # Prefer bays at row ends for easier maneuvering
                    if bay.position <= 2 or bay.position >= 18:
                        score -= 20
                    else:
                        score += size_penalty
                
                score += occupancy_penalty - priority_bonus
                
                if score < best_score:
                    best_score = score
                    best_bay = bay
        
        return best_bay.id if best_bay else None


ASSIGNMENT_STRATEGIES = {
    "random": RandomAssignment(),
    "greedy": GreedyAssignment(),
    "smart": SmartAssignment()
}


class NeuralNetworkAssignmentStrategy(AssignmentStrategy):
    """Assignment strategy using neural network."""
    
    def __init__(self):
        self._network = None
        self._nn_assignment = None
        self.last_confidence: float = 0.0
    
    @property
    def network(self):
        """Lazy load neural network."""
        if self._network is None:
            from ..neural_network import NeuralNetwork, NeuralNetworkAssignment
            self._network = NeuralNetwork()
            self._nn_assignment = NeuralNetworkAssignment(self._network)
        return self._network
    
    @network.setter
    def network(self, value):
        """Set the neural network."""
        self._network = value
        if value is not None:
            from ..neural_network import NeuralNetworkAssignment
            self._nn_assignment = NeuralNetworkAssignment(value)
    
    def assign(self, vehicle: Vehicle, car_park: CarPark, shops: List[Shop],
               shop_floor_map: Dict[str, int], current_time: float = 0.0,
               queue_length: int = 0) -> Optional[str]:
        """Assign using neural network."""
        if self._nn_assignment is None:
            _ = self.network  # Trigger lazy load
        
        result = self._nn_assignment.assign(
            vehicle, car_park, shops, shop_floor_map, 
            current_time, queue_length
        )
        
        self.last_confidence = self._nn_assignment.last_prediction_info.get('confidence', 0.0)
        return result


# Create singleton for NN strategy
_nn_strategy = NeuralNetworkAssignmentStrategy()


def get_nn_strategy() -> NeuralNetworkAssignmentStrategy:
    """Get the neural network assignment strategy singleton."""
    return _nn_strategy


ASSIGNMENT_STRATEGIES["neural_network"] = _nn_strategy


@dataclass
class SimulationEngine:
    """Main simulation engine."""
    
    car_park: CarPark = field(default_factory=lambda: CarPark(id="carpark_1", name="Main Car Park"))
    shops: List[Shop] = field(default_factory=list)
    
    # Active vehicles (id -> Vehicle)
    active_vehicles: Dict[str, Vehicle] = field(default_factory=dict)
    
    # Departed vehicles history (limited to last 100)
    departed_vehicles: List[Vehicle] = field(default_factory=list)
    
    # Waiting queue
    waiting_queue: List[Vehicle] = field(default_factory=list)
    
    # Simulation state
    current_time: float = 0.0  # In minutes from simulation start
    is_running: bool = False
    
    # Configuration
    arrival_rate: float = 2.0  # Vehicles per minute
    assignment_strategy: str = "smart"
    
    # Stats
    stats: SimulationStats = field(default_factory=SimulationStats)

    # Traffic management
    entry_queue: list = field(default_factory=list)    # List[(Vehicle, queued_at)]
    in_transit: dict = field(default_factory=dict)      # vid -> {vehicle, eta, bay_id, level, zones}
    zone_occupancy: dict = field(default_factory=dict)  # zone_id -> current count
    _pending_animations: list = field(default_factory=list)  # consumed by app.py each step
    _entry_credits: float = 0.0                         # fractional entry slots per step

    # Cached lookups (rebuilt on shop regeneration)
    _shop_floor_map: Dict[str, int] = field(default_factory=dict)
    _shop_names: List[str] = field(default_factory=list)
    _floor_list: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.shops:
            floors = [level.level_number for level in self.car_park.levels]
            self.shops = generate_random_shops(floors, shops_per_floor=6)
        self._rebuild_caches()
        self._init_traffic()
    
    def _rebuild_caches(self):
        """Rebuild cached lookups for performance."""
        self._shop_floor_map = {shop.name: shop.floor for shop in self.shops}
        self._shop_floor_map.update({shop.id: shop.floor for shop in self.shops})
        self._shop_names = [shop.name for shop in self.shops]
        self._floor_list = [level.level_number for level in self.car_park.levels]
        
        # Assign shops to levels
        for level in self.car_park.levels:
            level.nearby_shops = [
                shop.name for shop in self.shops 
                if shop.floor == level.level_number
            ]
    
    def reset(self):
        """Reset simulation to initial state."""
        self.car_park.reset()
        self.active_vehicles.clear()
        self.departed_vehicles.clear()
        self.waiting_queue.clear()
        self.current_time = 0.0
        self.is_running = False
        self.stats = SimulationStats()
        self._init_traffic()

        floors = [level.level_number for level in self.car_park.levels]
        self.shops = generate_random_shops(floors, shops_per_floor=6)
        self._rebuild_caches()
    
    def generate_arrival(self) -> Vehicle:
        """Generate a new vehicle arrival."""
        return Vehicle.generate_random(
            current_time=self.current_time,
            shop_list=self._shop_names,
            floor_list=self._floor_list
        )
    
    def _compute_bay_fit(self, vehicle: Vehicle, bay, assigned_level: int) -> float:
        """
        0-1 score measuring how well this bay suits this specific vehicle.

        Components (weighted sum):
          - floor proximity  (0.35): assigned level vs vehicle's target floor
          - type match       (0.40): correct bay type for vehicle's requirements;
                                     penalises wasting special bays on standard vehicles
          - lift proximity   (0.25): closeness to shop gate (scaled by visit purpose)
        """
        # Floor proximity
        target = vehicle.target_floor if vehicle.target_floor is not None else 0
        floor_score = max(0.0, 1.0 - abs(assigned_level - target) / 2.0)

        # Bay type match
        bt = bay.bay_type
        if vehicle.requires_blue_badge:
            type_score = 1.0 if bt == BayType.BLUE_BADGE   else 0.1
        elif vehicle.requires_parent_child:
            type_score = 1.0 if bt == BayType.PARENT_CHILD else 0.1
        elif vehicle.requires_ev_charging:
            type_score = 1.0 if bt == BayType.EV           else 0.1
        elif bt == BayType.STANDARD:
            type_score = 1.0   # standard car in standard bay — ideal
        else:
            type_score = 0.0   # standard car wasting a reserved special bay

        # Lift/shop-gate proximity, weighted by visit purpose
        # Distances on the grid run roughly 90–480 px; normalise by 500.
        purpose_weight = {
            'shopping':      1.0,
            'dining':        0.9,
            'quick_errand':  1.0,
            'entertainment': 0.7,
            'medical':       0.9,
            'commute':       0.3,
            'emergency':     0.8,
        }.get(vehicle.purpose.value, 0.7)
        raw_lift = max(0.0, 1.0 - bay.distance_to_lift / 500.0)
        # Blend: high-purpose vehicles really want to be close; commuters don't mind
        lift_score = raw_lift * purpose_weight + 0.5 * (1.0 - purpose_weight)

        return floor_score * 0.35 + type_score * 0.40 + lift_score * 0.25

    def try_assign_parking(self, vehicle: Vehicle) -> bool:
        """Try to assign a bay for a vehicle. Returns True if bay assigned."""
        # Support per-instance NN strategy override for isolated parallel training evaluation
        if self.assignment_strategy == "neural_network" and hasattr(self, '_nn_strategy_override'):
            strategy = self._nn_strategy_override
        else:
            strategy = ASSIGNMENT_STRATEGIES.get(self.assignment_strategy, ASSIGNMENT_STRATEGIES["smart"])

        if self.assignment_strategy == "neural_network":
            bay_id = strategy.assign(
                vehicle, self.car_park, self.shops, self._shop_floor_map,
                self.current_time, len(self.waiting_queue)
            )
            if hasattr(strategy, 'last_confidence'):
                self.stats.nn_assignments += 1
                n = self.stats.nn_assignments
                self.stats.nn_avg_confidence = (
                    (self.stats.nn_avg_confidence * (n - 1) + strategy.last_confidence) / n
                )
        else:
            bay_id = strategy.assign(vehicle, self.car_park, self.shops, self._shop_floor_map)

        if bay_id:
            if self.car_park.assign_bay(vehicle.id, bay_id, self.current_time):
                vehicle.assigned_bay_id = bay_id
                try:
                    level_part = bay_id.split('_')[0]
                    vehicle.assigned_level = int(level_part[1:])
                except (IndexError, ValueError):
                    vehicle.assigned_level = 1

                self.stats.total_arrivals += 1
                self.stats.arrivals_by_purpose[vehicle.purpose.value] = \
                    self.stats.arrivals_by_purpose.get(vehicle.purpose.value, 0) + 1
                if vehicle.assigned_level is not None:
                    self.stats.arrivals_by_level[vehicle.assigned_level] = \
                        self.stats.arrivals_by_level.get(vehicle.assigned_level, 0) + 1

                # Track per-assignment goodness-of-fit
                bay = self.car_park.find_bay(bay_id)
                if bay is not None:
                    fit = self._compute_bay_fit(vehicle, bay, vehicle.assigned_level or 0)
                    n = self.stats.total_arrivals
                    self.stats.avg_assignment_quality = (
                        (self.stats.avg_assignment_quality * (n - 1) + fit) / n
                    )

                return True
        return False
    
    def process_arrival(self, vehicle: Vehicle):
        """Process a vehicle arrival."""
        if not self.try_assign_parking(vehicle):
            self.waiting_queue.append(vehicle)
            self.stats.total_denied += 1
        else:
            self._enter_or_queue(vehicle)
    
    def process_departure(self, vehicle_id: str):
        """Process a vehicle departure."""
        vehicle = self.active_vehicles.pop(vehicle_id, None)
        if not vehicle:
            return
            
        vehicle.actual_departure_time = self.current_time
        
        if vehicle.assigned_bay_id:
            self.car_park.release_bay(vehicle.assigned_bay_id)
        
        # Keep limited history
        self.departed_vehicles.append(vehicle)
        if len(self.departed_vehicles) > 100:
            self.departed_vehicles = self.departed_vehicles[-100:]
        
        self.stats.total_departures += 1
        
        # Update average stay (incremental calculation)
        stay_duration = self.current_time - vehicle.arrival_time
        n = self.stats.total_departures
        self.stats.avg_stay_duration = (
            (self.stats.avg_stay_duration * (n - 1) + stay_duration) / n
        )
        
        # Try to assign parking to waiting vehicles
        self._process_waiting_queue()
    
    def _process_waiting_queue(self):
        """Try to assign parking to waiting vehicles."""
        still_waiting = []
        for vehicle in self.waiting_queue:
            if self.try_assign_parking(vehicle):
                self._enter_or_queue(vehicle)
            else:
                still_waiting.append(vehicle)
        self.waiting_queue = still_waiting

    def _init_traffic(self):
        """Initialise traffic management state."""
        self._zone_capacity = {
            'entry_gate': 3,    # max cars in entry lane at once
            'ramp_1_0':   1,    # single-lane ramp level 1 → 0
            'ramp_0_m1':  1,    # single-lane ramp level 0 → -1
        }
        self.zone_occupancy = {k: 0 for k in self._zone_capacity}
        self._entry_throughput = 1.5   # vehicles/minute max through entry gate
        self._entry_credits = 0.0
        self.entry_queue = []
        self.in_transit = {}
        self._pending_animations = []

    def _enter_or_queue(self, vehicle: Vehicle):
        """Vehicle has an assigned bay — either start transit or join entry queue."""
        gate_occ = self.zone_occupancy.get('entry_gate', 0)
        gate_cap = self._zone_capacity.get('entry_gate', 3)
        if gate_occ < gate_cap:
            self._start_transit(vehicle)
        else:
            self.entry_queue.append((vehicle, self.current_time))
            self.stats.congestion_events += 1
            self.stats.peak_entry_queue = max(self.stats.peak_entry_queue, len(self.entry_queue))

    def _start_transit(self, vehicle: Vehicle):
        """Begin vehicle's physical journey to its bay."""
        level = vehicle.assigned_level if vehicle.assigned_level is not None else 1

        zones = ['entry_gate']
        if level < 1:
            zones.append('ramp_1_0')
        if level < 0:
            zones.append('ramp_0_m1')

        for z in zones:
            self.zone_occupancy[z] = self.zone_occupancy.get(z, 0) + 1

        travel_time = self._compute_travel_time(level, zones)
        eta = self.current_time + travel_time

        self.in_transit[vehicle.id] = {
            'vehicle': vehicle,
            'eta': eta,
            'bay_id': vehicle.assigned_bay_id,
            'level': level,
            'zones': zones,
            'started_at': self.current_time,
        }

        self._pending_animations.append({
            'type': 'enter',
            'vehicle_id': vehicle.id,
            'level': level,
            'bay_id': vehicle.assigned_bay_id,
            'timestamp': self.current_time,
        })

    def _compute_travel_time(self, level: int, zones: list) -> float:
        """Minutes from entry gate to bay, accounting for congestion."""
        base_times = {1: 1.2, 0: 2.5, -1: 4.0}
        base = base_times.get(level, 1.5)

        extra = 0.0
        for zone in zones:
            cap = self._zone_capacity.get(zone, 3)
            occ = self.zone_occupancy.get(zone, 0)
            if cap > 0 and occ >= cap:
                # Zone at/over capacity: add delay per queued car
                extra += 0.8 + max(0, occ - cap) * 0.5
            elif cap > 0 and occ > 0:
                extra += (occ / cap) * 0.25

        return base + extra

    def _process_entry_queue(self, time_delta: float):
        """Release vehicles from entry queue as gate capacity allows."""
        if not self.entry_queue:
            return

        self._entry_credits += self._entry_throughput * time_delta

        while self.entry_queue and self._entry_credits >= 1.0:
            gate_occ = self.zone_occupancy.get('entry_gate', 0)
            gate_cap = self._zone_capacity.get('entry_gate', 3)
            if gate_occ >= gate_cap:
                break  # Gate still full — wait for next step

            vehicle, queued_at = self.entry_queue.pop(0)
            entry_wait = self.current_time - queued_at

            total = max(1, self.stats.total_arrivals)
            self.stats.avg_entry_wait = (
                (self.stats.avg_entry_wait * (total - 1) + entry_wait) / total
            )

            self._start_transit(vehicle)
            self._entry_credits -= 1.0

    def _process_in_transit(self):
        """Move vehicles that have finished transit into active_vehicles."""
        arrived = [vid for vid, info in self.in_transit.items()
                   if self.current_time >= info['eta']]

        for vid in arrived:
            info = self.in_transit.pop(vid)
            vehicle = info['vehicle']

            self.car_park.finalize_bay(vid, info['bay_id'], self.current_time)
            self.active_vehicles[vid] = vehicle

            # Release zones
            for zone in info['zones']:
                self.zone_occupancy[zone] = max(0, self.zone_occupancy.get(zone, 0) - 1)

    def step(self, time_delta: float = 1.0):
        """Advance simulation by time_delta minutes."""
        self.current_time += time_delta

        # Generate arrivals (Poisson process)
        expected_arrivals = self.arrival_rate * time_delta
        num_arrivals = max(0, int(random.gauss(expected_arrivals, expected_arrivals * 0.3)))
        for _ in range(num_arrivals):
            vehicle = self.generate_arrival()
            self.process_arrival(vehicle)

        # Process entry queue (rate-limited gate)
        self._process_entry_queue(time_delta)

        # Advance in-transit vehicles
        self._process_in_transit()

        # Check for departures from active_vehicles
        to_depart = []
        for vehicle_id, vehicle in self.active_vehicles.items():
            stay_duration = self.current_time - vehicle.arrival_time
            departure_threshold = vehicle.estimated_stay_minutes * random.uniform(0.85, 1.15)
            if stay_duration >= departure_threshold:
                to_depart.append(vehicle_id)
        for vehicle_id in to_depart:
            self.process_departure(vehicle_id)

        # Update peak occupancy — track BAY COUNT, not ratio, so the fitness
        # formula (min(peak_occupancy, 200) * 20) works as intended.
        self.stats.peak_occupancy = max(self.stats.peak_occupancy, self.car_park.total_occupied)
    
    def get_state_summary(self) -> Dict:
        """Get current simulation state summary."""
        return {
            "current_time": self.current_time,
            "total_capacity": self.car_park.total_capacity,
            "total_occupied": self.car_park.total_occupied,
            "total_available": self.car_park.total_available,
            "occupancy_rate": self.car_park.overall_occupancy,
            "active_vehicles": len(self.active_vehicles),
            "waiting_queue": len(self.waiting_queue),
            "total_arrivals": self.stats.total_arrivals,
            "total_departures": self.stats.total_departures,
            "total_denied": self.stats.total_denied,
            "peak_occupancy": self.stats.peak_occupancy,
            # Traffic
            "entry_queue_length": len(self.entry_queue),
            "in_transit_count": len(self.in_transit),
            "zone_occupancy": dict(self.zone_occupancy),
            "zone_capacity": dict(self._zone_capacity) if hasattr(self, '_zone_capacity') else {},
            "avg_entry_wait": round(self.stats.avg_entry_wait, 1),
            "congestion_events": self.stats.congestion_events,
        }
    
    def get_level_summary(self, level_number: int) -> Optional[Dict]:
        """Get summary for a specific level."""
        level = self.car_park.get_level(level_number)
        if not level:
            return None
        
        return {
            "level_number": level.level_number,
            "name": level.name,
            "capacity": level.capacity,
            "occupied": level.occupied_count,
            "available": level.available_count,
            "occupancy_rate": level.occupancy_rate,
            "nearby_shops": level.nearby_shops
        }
