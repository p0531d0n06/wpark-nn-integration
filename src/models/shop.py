"""Shop data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import random


class ShopCategory(Enum):
    FASHION = "fashion"
    ELECTRONICS = "electronics"
    FOOD = "food"
    SERVICES = "services"
    ENTERTAINMENT = "entertainment"
    GROCERY = "grocery"
    HEALTH = "health"
    HOME = "home"


# Sample shop names by category for random generation
SHOP_NAMES = {
    ShopCategory.FASHION: [
        "Style Hub", "Urban Threads", "Fashion Forward", "Trend Setter",
        "Classic Wear", "Modern Fit", "Chic Boutique", "Street Style",
        "Elegant Edge", "Casual Corner"
    ],
    ShopCategory.ELECTRONICS: [
        "Tech Zone", "Digital World", "Gadget Galaxy", "Smart Store",
        "Electro Hub", "Device Den", "Circuit City", "Pixel Palace"
    ],
    ShopCategory.FOOD: [
        "Cafe Central", "Bistro Blue", "Quick Bites", "Gourmet Garden",
        "Food Factory", "Tasty Treats", "Deli Delight", "Snack Shack",
        "Pizza Palace", "Noodle House"
    ],
    ShopCategory.SERVICES: [
        "Quick Fix", "Service Center", "Express Aid", "Help Hub",
        "Solution Spot", "Care Corner"
    ],
    ShopCategory.ENTERTAINMENT: [
        "Fun Zone", "Play Palace", "Game Galaxy", "Adventure Arcade",
        "Cinema Plus", "Bowl & Roll"
    ],
    ShopCategory.GROCERY: [
        "Fresh Market", "Daily Mart", "Green Grocer", "Super Store",
        "Mini Mart", "Farm Fresh"
    ],
    ShopCategory.HEALTH: [
        "Health Hub", "Wellness Center", "Pharma Plus", "Fit First",
        "Care Clinic", "Vita Zone"
    ],
    ShopCategory.HOME: [
        "Home Haven", "Decor Plus", "Furniture Flair", "Interior Ideas",
        "Cozy Corner", "Living Space"
    ]
}

# Typical stay duration (minutes) by category
STAY_DURATION_BY_CATEGORY = {
    ShopCategory.FASHION: (20, 60),
    ShopCategory.ELECTRONICS: (30, 90),
    ShopCategory.FOOD: (30, 75),
    ShopCategory.SERVICES: (15, 45),
    ShopCategory.ENTERTAINMENT: (60, 180),
    ShopCategory.GROCERY: (20, 60),
    ShopCategory.HEALTH: (15, 60),
    ShopCategory.HOME: (30, 90)
}


@dataclass
class Shop:
    """Shop in the mall connected to car park."""
    id: str
    name: str
    category: ShopCategory
    floor: int  # Mall floor (maps to car park level)
    
    # Location within floor (for distance calculations)
    x_position: float = 0.0
    y_position: float = 0.0
    
    # Popularity affects arrival rates
    popularity: float = 0.5  # 0.0 - 1.0
    
    # Typical visit duration
    avg_visit_minutes: float = 45.0
    
    def get_estimated_stay(self) -> float:
        """Get randomized estimated stay duration."""
        min_stay, max_stay = STAY_DURATION_BY_CATEGORY.get(
            self.category, (30, 60)
        )
        return random.uniform(min_stay, max_stay)
    
    def get_display_color(self) -> str:
        """Get color for display based on category."""
        colors = {
            ShopCategory.FASHION: "#FF69B4",
            ShopCategory.ELECTRONICS: "#4169E1",
            ShopCategory.FOOD: "#FFD700",
            ShopCategory.SERVICES: "#808080",
            ShopCategory.ENTERTAINMENT: "#FF6347",
            ShopCategory.GROCERY: "#32CD32",
            ShopCategory.HEALTH: "#00CED1",
            ShopCategory.HOME: "#DEB887"
        }
        return colors.get(self.category, "#CCCCCC")
    
    @staticmethod
    def generate_random(shop_id: str, floor: int) -> 'Shop':
        """Generate a random shop for a given floor."""
        category = random.choice(list(ShopCategory))
        names = SHOP_NAMES.get(category, ["Unknown Shop"])
        name = random.choice(names)
        
        return Shop(
            id=shop_id,
            name=name,
            category=category,
            floor=floor,
            x_position=random.uniform(0, 100),
            y_position=random.uniform(0, 50),
            popularity=random.uniform(0.3, 1.0),
            avg_visit_minutes=sum(STAY_DURATION_BY_CATEGORY.get(category, (30, 60))) / 2
        )


def generate_random_shops(floors: List[int], shops_per_floor: int = 6) -> List[Shop]:
    """Generate random shops for given floors."""
    shops = []
    shop_counter = 0
    
    for floor in floors:
        for i in range(shops_per_floor):
            shop_counter += 1
            shop = Shop.generate_random(f"shop_{shop_counter:03d}", floor)
            shops.append(shop)
    
    return shops
