"""Simulation engine for car park operations."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import random

from ..models.car_park import CarPark, Level, ParkingBay, BayType, BayStatus
from ..models.vehicle import Vehicle, VehicleSize, VisitPurpose, MobilityLevel
from ..models.shop import Shop, generate_random_shops


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
    """Smart assignment considering multiple factors with optimizations."""
    
    def assign(self, vehicle: Vehicle, car_park: CarPark, shops: List[Shop],
               shop_floor_map: Dict[str, int]) -> Optional[str]:
        target_floor = self._get_target_floor(vehicle, shop_floor_map)
        
        # Sort levels by distance for early termination
        sorted_levels = sorted(car_park.levels,
                               key=lambda l: abs(l.level_number - target_floor))
        
        best_bay = None
        best_score = float('inf')
        
        # Pre-check vehicle requirements
        needs_blue_badge = vehicle.requires_blue_badge
        needs_parent_child = vehicle.requires_parent_child
        is_ev = vehicle.is_ev
        priority_bonus = vehicle.priority * 30
        
        for level in sorted_levels:
            floor_distance = abs(level.level_number - target_floor)
            base_floor_score = floor_distance * 100
            
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
                
                score += bay.distance_to_lift * 0.5 + occupancy_penalty - priority_bonus
                
                if score < best_score:
                    best_score = score
                    best_bay = bay
        
        return best_bay.id if best_bay else None


ASSIGNMENT_STRATEGIES = {
    "random": RandomAssignment(),
    "greedy": GreedyAssignment(),
    "smart": SmartAssignment()
}


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
    
    # Cached lookups (rebuilt on shop regeneration)
    _shop_floor_map: Dict[str, int] = field(default_factory=dict)
    _shop_names: List[str] = field(default_factory=list)
    _floor_list: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.shops:
            floors = [level.level_number for level in self.car_park.levels]
            self.shops = generate_random_shops(floors, shops_per_floor=6)
        self._rebuild_caches()
    
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
        
        # Regenerate shops
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
    
    def try_assign_parking(self, vehicle: Vehicle) -> bool:
        """Try to assign parking to a vehicle."""
        strategy = ASSIGNMENT_STRATEGIES.get(self.assignment_strategy, ASSIGNMENT_STRATEGIES["smart"])
        bay_id = strategy.assign(vehicle, self.car_park, self.shops, self._shop_floor_map)
        
        if bay_id:
            if self.car_park.assign_bay(vehicle.id, bay_id, self.current_time):
                vehicle.assigned_bay_id = bay_id
                
                # Extract level from bay_id (format: L{level}_{row}{pos})
                try:
                    level_part = bay_id.split('_')[0]  # e.g., "L-1" or "L0"
                    vehicle.assigned_level = int(level_part[1:])
                except (IndexError, ValueError):
                    vehicle.assigned_level = 0
                
                self.active_vehicles[vehicle.id] = vehicle
                self.stats.total_arrivals += 1
                
                # Update stats
                self.stats.arrivals_by_purpose[vehicle.purpose.value] = \
                    self.stats.arrivals_by_purpose.get(vehicle.purpose.value, 0) + 1
                
                if vehicle.assigned_level is not None:
                    self.stats.arrivals_by_level[vehicle.assigned_level] = \
                        self.stats.arrivals_by_level.get(vehicle.assigned_level, 0) + 1
                
                return True
        
        return False
    
    def process_arrival(self, vehicle: Vehicle):
        """Process a vehicle arrival."""
        if not self.try_assign_parking(vehicle):
            self.waiting_queue.append(vehicle)
            self.stats.total_denied += 1
    
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
            if not self.try_assign_parking(vehicle):
                still_waiting.append(vehicle)
        self.waiting_queue = still_waiting
    
    def step(self, time_delta: float = 1.0):
        """Advance simulation by time_delta minutes."""
        self.current_time += time_delta
        
        # Generate arrivals (Poisson process)
        expected_arrivals = self.arrival_rate * time_delta
        num_arrivals = max(0, int(random.gauss(expected_arrivals, expected_arrivals * 0.3)))
        
        for _ in range(num_arrivals):
            vehicle = self.generate_arrival()
            self.process_arrival(vehicle)
        
        # Check for departures
        to_depart = []
        for vehicle_id, vehicle in self.active_vehicles.items():
            stay_duration = self.current_time - vehicle.arrival_time
            
            # Add some randomness to departure
            departure_threshold = vehicle.estimated_stay_minutes * random.uniform(0.8, 1.2)
            
            if stay_duration >= departure_threshold:
                to_depart.append(vehicle_id)
        
        for vehicle_id in to_depart:
            self.process_departure(vehicle_id)
        
        # Update peak occupancy
        current_occupancy = self.car_park.overall_occupancy
        self.stats.peak_occupancy = max(self.stats.peak_occupancy, current_occupancy)
    
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
            "peak_occupancy": self.stats.peak_occupancy
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
