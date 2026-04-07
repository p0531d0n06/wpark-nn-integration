"""Vehicle data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import random
import uuid


class VehicleSize(Enum):
    COMPACT = "compact"
    STANDARD = "standard"
    LARGE = "large"  # SUVs, vans
    OVERSIZED = "oversized"  # Trucks, RVs


class VisitPurpose(Enum):
    SHOPPING = "shopping"
    DINING = "dining"
    ENTERTAINMENT = "entertainment"
    MEDICAL = "medical"
    EMERGENCY = "emergency"
    QUICK_ERRAND = "quick_errand"
    COMMUTE = "commute"


class MobilityLevel(Enum):
    FULL = "full"
    LIMITED = "limited"  # Needs closer parking
    WHEELCHAIR = "wheelchair"  # Requires blue badge
    ELDERLY = "elderly"
    WITH_CHILDREN = "with_children"  # Parent/child bay


@dataclass
class Vehicle:
    """Vehicle entering the car park."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Vehicle properties
    size: VehicleSize = VehicleSize.STANDARD
    is_ev: bool = False
    
    # Driver properties
    purpose: VisitPurpose = VisitPurpose.SHOPPING
    mobility: MobilityLevel = MobilityLevel.FULL
    
    # Target destination
    target_shop: Optional[str] = None
    target_floor: Optional[int] = None
    
    # Timing
    arrival_time: float = 0.0
    estimated_stay_minutes: float = 60.0
    actual_departure_time: Optional[float] = None
    
    # Assignment
    assigned_bay_id: Optional[str] = None
    assigned_level: Optional[int] = None
    
    # Priority (0-1, higher = more urgent)
    priority: float = 0.5
    
    def __post_init__(self):
        # Set priority based on purpose and mobility
        self._calculate_priority()
    
    def _calculate_priority(self):
        """Calculate priority based on purpose and mobility."""
        base_priority = 0.5
        
        # Purpose modifiers
        purpose_priority = {
            VisitPurpose.EMERGENCY: 1.0,
            VisitPurpose.MEDICAL: 0.8,
            VisitPurpose.QUICK_ERRAND: 0.6,
            VisitPurpose.SHOPPING: 0.5,
            VisitPurpose.DINING: 0.5,
            VisitPurpose.ENTERTAINMENT: 0.4,
            VisitPurpose.COMMUTE: 0.3
        }
        
        # Mobility modifiers
        mobility_priority = {
            MobilityLevel.WHEELCHAIR: 0.9,
            MobilityLevel.ELDERLY: 0.7,
            MobilityLevel.LIMITED: 0.6,
            MobilityLevel.WITH_CHILDREN: 0.6,
            MobilityLevel.FULL: 0.5
        }
        
        self.priority = max(
            purpose_priority.get(self.purpose, 0.5),
            mobility_priority.get(self.mobility, 0.5)
        )
    
    @property
    def requires_blue_badge(self) -> bool:
        return self.mobility == MobilityLevel.WHEELCHAIR
    
    @property
    def requires_parent_child(self) -> bool:
        return self.mobility == MobilityLevel.WITH_CHILDREN
    
    @property
    def requires_ev_charging(self) -> bool:
        return self.is_ev
    
    def get_display_color(self) -> str:
        """Get display color based on vehicle properties."""
        if self.purpose == VisitPurpose.EMERGENCY:
            return "#FF0000"
        elif self.requires_blue_badge:
            return "#4169E1"
        elif self.requires_parent_child:
            return "#DDA0DD"
        elif self.is_ev:
            return "#32CD32"
        elif self.purpose == VisitPurpose.SHOPPING:
            return "#FFB347"
        elif self.purpose == VisitPurpose.DINING:
            return "#87CEEB"
        else:
            return "#808080"
    
    @staticmethod
    def generate_random(current_time: float = 0.0, shop_list: List[str] = None, floor_list: List[int] = None) -> 'Vehicle':
        """Generate a random vehicle."""
        vehicle = Vehicle()
        
        # Random size (biased toward standard)
        vehicle.size = random.choices(
            list(VehicleSize),
            weights=[20, 55, 20, 5]
        )[0]
        
        # Random EV (15% chance)
        vehicle.is_ev = random.random() < 0.15
        
        # Random purpose
        vehicle.purpose = random.choices(
            list(VisitPurpose),
            weights=[40, 20, 15, 5, 2, 10, 8]  # Shopping most common
        )[0]
        
        # Random mobility (mostly full)
        vehicle.mobility = random.choices(
            list(MobilityLevel),
            weights=[75, 8, 5, 5, 7]
        )[0]
        
        # Target shop/floor
        if shop_list:
            vehicle.target_shop = random.choice(shop_list)
        if floor_list:
            vehicle.target_floor = random.choice(floor_list)
        
        # Arrival time
        vehicle.arrival_time = current_time
        
        # Estimated stay based on purpose
        stay_by_purpose = {
            VisitPurpose.EMERGENCY: random.randint(5, 30),
            VisitPurpose.QUICK_ERRAND: random.randint(10, 30),
            VisitPurpose.MEDICAL: random.randint(30, 90),
            VisitPurpose.SHOPPING: random.randint(30, 180),
            VisitPurpose.DINING: random.randint(45, 120),
            VisitPurpose.ENTERTAINMENT: random.randint(60, 180),
            VisitPurpose.COMMUTE: random.randint(240, 540)
        }
        vehicle.estimated_stay_minutes = stay_by_purpose.get(vehicle.purpose, 60)
        
        vehicle._calculate_priority()
        return vehicle
