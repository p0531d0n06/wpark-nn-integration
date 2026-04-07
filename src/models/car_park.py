"""Car park data models - 3 floor simplified structure."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import random


class BayType(Enum):
    STANDARD = "standard"
    BLUE_BADGE = "blue_badge"
    PARENT_CHILD = "parent_child"
    EV = "ev"


class BayStatus(Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"


class BaySize(Enum):
    COMPACT = "compact"
    STANDARD = "standard"
    LARGE = "large"


@dataclass
class ParkingBay:
    """Individual parking bay."""
    id: str
    level_id: str
    row: str
    position: int
    bay_type: BayType = BayType.STANDARD
    status: BayStatus = BayStatus.AVAILABLE
    size: BaySize = BaySize.STANDARD
    
    # Position (grid coordinates)
    x: float = 0.0
    y: float = 0.0
    
    # Distances (calculated)
    distance_to_lift: float = 0.0
    distance_to_exit: float = 0.0
    
    # Occupancy tracking
    occupied_by: Optional[str] = None  # Vehicle ID
    occupied_since: Optional[float] = None  # Timestamp
    
    def occupy(self, vehicle_id: str, timestamp: float):
        """Mark bay as occupied."""
        self.status = BayStatus.OCCUPIED
        self.occupied_by = vehicle_id
        self.occupied_since = timestamp
    
    def vacate(self):
        """Mark bay as available."""
        self.status = BayStatus.AVAILABLE
        self.occupied_by = None
        self.occupied_since = None
    
    @property
    def is_available(self) -> bool:
        return self.status == BayStatus.AVAILABLE
    
    def get_color(self) -> str:
        """Get display color based on type and status."""
        if self.status == BayStatus.OCCUPIED:
            return "#FF6B6B"  # Red
        elif self.status == BayStatus.RESERVED:
            return "#FFD93D"  # Yellow
        elif self.status == BayStatus.MAINTENANCE:
            return "#6C757D"  # Grey
        elif self.bay_type == BayType.BLUE_BADGE:
            return "#4169E1"  # Blue
        elif self.bay_type == BayType.PARENT_CHILD:
            return "#DDA0DD"  # Purple
        elif self.bay_type == BayType.EV:
            return "#32CD32"  # Green
        else:
            return "#90EE90"  # Light green (available standard)


@dataclass
class Level:
    """Car park level/floor."""
    id: str
    level_number: int  # -1, 0, 1, 2 etc.
    name: str
    rows: int = 4
    bays_per_row: int = 20
    
    # Connected mall floor info
    connected_mall_floor: str = ""
    nearby_shops: List[str] = field(default_factory=list)
    
    # Bays storage
    bays: List[ParkingBay] = field(default_factory=list)
    
    # Special bay counts
    blue_badge_count: int = 4
    parent_child_count: int = 4
    ev_count: int = 2
    
    # Layout dimensions
    width: float = 100.0
    height: float = 60.0
    
    def __post_init__(self):
        if not self.bays:
            self._generate_bays()
    
    def _generate_bays(self):
        """Generate parking bays for this level."""
        self.bays = []
        row_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:self.rows]
        
        bay_width = 4.0
        bay_height = 8.0
        aisle_height = 12.0
        margin = 10.0
        
        special_bays_assigned = {
            'blue_badge': 0,
            'parent_child': 0,
            'ev': 0
        }
        
        for row_idx, row_label in enumerate(row_labels):
            y = margin + row_idx * (bay_height + aisle_height)
            
            for pos in range(self.bays_per_row):
                x = margin + pos * bay_width
                
                # Determine bay type
                bay_type = BayType.STANDARD
                
                # Place special bays near the lift (first few positions of first rows)
                if row_idx == 0 and pos < self.blue_badge_count and special_bays_assigned['blue_badge'] < self.blue_badge_count:
                    bay_type = BayType.BLUE_BADGE
                    special_bays_assigned['blue_badge'] += 1
                elif row_idx == 1 and pos < self.parent_child_count and special_bays_assigned['parent_child'] < self.parent_child_count:
                    bay_type = BayType.PARENT_CHILD
                    special_bays_assigned['parent_child'] += 1
                elif row_idx == 0 and pos >= self.bays_per_row - self.ev_count and special_bays_assigned['ev'] < self.ev_count:
                    bay_type = BayType.EV
                    special_bays_assigned['ev'] += 1
                
                bay = ParkingBay(
                    id=f"L{self.level_number}_{row_label}{pos+1:02d}",
                    level_id=self.id,
                    row=row_label,
                    position=pos + 1,
                    bay_type=bay_type,
                    x=x,
                    y=y,
                    distance_to_lift=x + y * 0.5,  # Simplified distance calc
                    distance_to_exit=abs(x - 50) + y
                )
                self.bays.append(bay)
    
    @property
    def capacity(self) -> int:
        return len(self.bays)
    
    @property
    def occupied_count(self) -> int:
        return sum(1 for bay in self.bays if bay.status == BayStatus.OCCUPIED)
    
    @property
    def available_count(self) -> int:
        return sum(1 for bay in self.bays if bay.status == BayStatus.AVAILABLE)
    
    @property
    def occupancy_rate(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.occupied_count / self.capacity
    
    def get_available_bays(self, bay_type: Optional[BayType] = None, 
                           min_size: Optional[BaySize] = None) -> List[ParkingBay]:
        """Get list of available bays, optionally filtered."""
        available = [b for b in self.bays if b.is_available]
        
        if bay_type:
            available = [b for b in available if b.bay_type == bay_type]
        
        return available
    
    def get_bay_by_id(self, bay_id: str) -> Optional[ParkingBay]:
        """Find bay by ID."""
        for bay in self.bays:
            if bay.id == bay_id:
                return bay
        return None


@dataclass
class CarPark:
    """Complete car park with multiple levels."""
    id: str
    name: str
    levels: List[Level] = field(default_factory=list)
    
    # Entry/exit info
    entry_level: int = 0
    exit_level: int = 0
    
    def __post_init__(self):
        if not self.levels:
            self._generate_default_levels()
    
    def _generate_default_levels(self):
        """Generate 3 default levels."""
        self.levels = [
            Level(
                id="level_-1",
                level_number=-1,
                name="Level -1 (Underground)",
                rows=4,
                bays_per_row=15,
                connected_mall_floor="Lower Ground",
                blue_badge_count=4,
                parent_child_count=3,
                ev_count=2
            ),
            Level(
                id="level_0",
                level_number=0,
                name="Level 0 (Ground)",
                rows=4,
                bays_per_row=18,
                connected_mall_floor="Ground Floor",
                blue_badge_count=6,
                parent_child_count=4,
                ev_count=3
            ),
            Level(
                id="level_1",
                level_number=1,
                name="Level 1 (First Floor)",
                rows=4,
                bays_per_row=20,
                connected_mall_floor="First Floor",
                blue_badge_count=4,
                parent_child_count=3,
                ev_count=2
            )
        ]
    
    @property
    def total_capacity(self) -> int:
        return sum(level.capacity for level in self.levels)
    
    @property
    def total_occupied(self) -> int:
        return sum(level.occupied_count for level in self.levels)
    
    @property
    def total_available(self) -> int:
        return sum(level.available_count for level in self.levels)
    
    @property
    def overall_occupancy(self) -> float:
        if self.total_capacity == 0:
            return 0.0
        return self.total_occupied / self.total_capacity
    
    def get_level(self, level_number: int) -> Optional[Level]:
        """Get level by number."""
        for level in self.levels:
            if level.level_number == level_number:
                return level
        return None
    
    def get_all_available_bays(self, bay_type: Optional[BayType] = None) -> List[ParkingBay]:
        """Get all available bays across all levels."""
        all_bays = []
        for level in self.levels:
            all_bays.extend(level.get_available_bays(bay_type))
        return all_bays
    
    def find_bay(self, bay_id: str) -> Optional[ParkingBay]:
        """Find a bay by ID across all levels."""
        for level in self.levels:
            bay = level.get_bay_by_id(bay_id)
            if bay:
                return bay
        return None
    
    def assign_bay(self, vehicle_id: str, bay_id: str, timestamp: float) -> bool:
        """Assign a vehicle to a bay."""
        bay = self.find_bay(bay_id)
        if bay and bay.is_available:
            bay.occupy(vehicle_id, timestamp)
            return True
        return False
    
    def release_bay(self, bay_id: str) -> bool:
        """Release a bay."""
        bay = self.find_bay(bay_id)
        if bay:
            bay.vacate()
            return True
        return False
    
    def reset(self):
        """Reset all bays to available."""
        for level in self.levels:
            for bay in level.bays:
                bay.vacate()
