# WPARK Neural Network Car Park Simulation

## Project Plan: Genetic Neural Network for Intelligent Parking Assignment

**Version:** 1.0  
**Created:** 2026-04-07  
**Status:** Planning Phase

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Objectives](#2-project-objectives)
3. [System Architecture](#3-system-architecture)
4. [Data Models](#4-data-models)
5. [Neural Network Design](#5-neural-network-design)
6. [Genetic Algorithm Training](#6-genetic-algorithm-training)
7. [Simulation Engine](#7-simulation-engine)
8. [Streamlit UI Design](#8-streamlit-ui-design)
9. [Implementation Phases](#9-implementation-phases)
10. [Technical Requirements](#10-technical-requirements)
11. [Success Metrics](#11-success-metrics)
12. [Risk Assessment](#12-risk-assessment)

---

## 1. Executive Summary

This project develops an intelligent car park assignment system using a **Genetic Neural Network (GNN)** trained through simulation. The system optimizes parking bay allocation based on multiple factors including purpose of visit, destination proximity, predicted stay duration, vehicle characteristics, and accessibility requirements.

### Key Innovation
Traditional parking systems assign bays on a first-come-first-served basis. Our approach uses predictive modelling to:
- Reduce patron walking distance by 40-60%
- Decrease internal traffic congestion by 30-50%
- Improve turnover efficiency for high-demand areas
- Ensure accessibility compliance and prioritization
- Optimize emergency vehicle access and response times

---

## 2. Project Objectives

### 2.1 Primary Goals

| Goal | Description | Target Metric |
|------|-------------|---------------|
| **Efficient Assignment** | Assign optimal parking bay based on multiple criteria | 85%+ optimal assignments |
| **Reduce Congestion** | Minimize internal car park traffic | 40% reduction in avg travel time |
| **Predictive Stay Duration** | Accurately predict parking duration | ±15 min accuracy for 80% of cases |
| **Accessibility Priority** | Ensure accessible bays used appropriately | 95%+ compliance |
| **Emergency Response** | Reserve/clear bays for emergency vehicles | <30 sec assignment time |

### 2.2 Secondary Goals

- Create reusable simulation framework for other car parks
- Generate training data for future ML improvements
- Provide real-time visualization of car park state
- Enable scenario testing (events, peak times, emergencies)

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT WEB UI                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Dashboard  │  │ Simulation  │  │  Training   │  │  Analytics  │ │
│  │   Control   │  │    View     │  │   Monitor   │  │   Reports   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│                       APPLICATION LAYER                              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Simulation Engine                             ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ ││
│  │  │  Event   │  │   Car    │  │  Traffic │  │   Time/Clock     │ ││
│  │  │ Generator│  │ Spawner  │  │  Router  │  │   Controller     │ ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                 Neural Network Module                            ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ ││
│  │  │ Feature  │  │   GNN    │  │  Output  │  │   Assignment     │ ││
│  │  │ Encoder  │  │  Model   │  │ Decoder  │  │   Executor       │ ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                 Genetic Algorithm Trainer                        ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ ││
│  │  │Population│  │ Fitness  │  │Selection │  │   Crossover/     │ ││
│  │  │ Manager  │  │ Evaluator│  │ Strategy │  │   Mutation       │ ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────┘│
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Car Park   │  │   Vehicle   │  │  Training   │  │   Model     │ │
│  │   Config    │  │    Queue    │  │    Data     │  │  Checkpoints│ │
│  │  (JSON/DB)  │  │  (Memory)   │  │  (SQLite)   │  │   (Files)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Interactions

```
Car Arrival → Feature Extraction → Neural Network → Bay Scoring → Assignment
                    ↓                    ↑
              Fitness Eval ←── Genetic Algorithm ←── Training Loop
                    ↓
            Model Update → Improved Predictions
```

---

## 4. Data Models

### 4.1 Car Park Structure

```python
@dataclass
class CarPark:
    id: str
    name: str
    levels: List[Level]
    entry_points: List[EntryPoint]
    exit_points: List[ExitPoint]
    lifts: List[Lift]
    total_capacity: int
    
@dataclass
class Level:
    id: str
    level_number: int  # -1, 0, 1, 2, 3, 4
    name: str
    rows: List[Row]
    aisles: List[Aisle]
    ramps: List[Ramp]
    facilities: List[Facility]
    connected_mall_floor: str
    nearby_shops: List[Shop]
    
@dataclass
class ParkingBay:
    id: str
    level_id: str
    row: str
    position: int
    bay_type: BayType  # STANDARD, BLUE_BADGE, PARENT_CHILD, EV, MOTORCYCLE
    status: BayStatus  # AVAILABLE, OCCUPIED, RESERVED, MAINTENANCE
    size: BaySize      # COMPACT, STANDARD, LARGE
    
    # Spatial properties
    x: float
    y: float
    distance_to_lift: float
    distance_to_stairs: float
    distance_to_ramp: float
    distance_to_entry: float
    distance_to_exit: float
    
    # Computed properties
    accessibility_score: float  # 0-1, higher = more accessible
    convenience_score: float    # 0-1, based on location
    turnover_rate: float        # Historical avg stays per day
```

### 4.2 Vehicle Model

```python
@dataclass
class Vehicle:
    id: str
    arrival_time: datetime
    
    # Physical properties
    size: VehicleSize          # COMPACT, STANDARD, LARGE, VAN
    is_ev: bool
    license_plate: str
    
    # Driver properties
    has_blue_badge: bool
    has_children: bool
    mobility_requirements: MobilityLevel  # FULL, LIMITED, WHEELCHAIR
    
    # Visit properties
    purpose: VisitPurpose
    destination_shops: List[str]
    estimated_stay: timedelta
    is_emergency: bool
    priority_level: int        # 0=normal, 10=max priority
    
    # Predicted/Actual
    predicted_stay: timedelta
    actual_departure: datetime = None
    assigned_bay: str = None

class VisitPurpose(Enum):
    SHOPPING_QUICK = "shopping_quick"      # <30 mins
    SHOPPING_REGULAR = "shopping_regular"  # 30-120 mins
    SHOPPING_EXTENDED = "shopping_extended"# 2-4 hours
    DINING = "dining"                      # 1-2 hours
    CINEMA = "cinema"                      # 2-3 hours
    GROCERY = "grocery"                    # 30-60 mins
    APPOINTMENT = "appointment"            # Variable
    WORK_EMPLOYEE = "work_employee"        # 4-9 hours
    EMERGENCY_MEDICAL = "emergency_medical"# Immediate
    EMERGENCY_SECURITY = "emergency_security"
    DELIVERY = "delivery"                  # 15-30 mins
    PICKUP_COLLECTION = "pickup_collection"# 5-15 mins
```

### 4.3 Shop/Destination Model

```python
@dataclass
class Shop:
    id: str
    name: str
    mall_floor: int
    category: ShopCategory
    x: float
    y: float
    avg_visit_duration: timedelta
    peak_hours: List[int]
    
class ShopCategory(Enum):
    ANCHOR_STORE = "anchor"        # John Lewis, Waitrose
    FASHION = "fashion"            # H&M, Zara
    ELECTRONICS = "electronics"    # Apple
    FOOD_DRINK = "food_drink"      # Costa, Pret
    SERVICES = "services"          # Banks, phone shops
    ENTERTAINMENT = "entertainment"# Cinema
    SUPERMARKET = "supermarket"
```

### 4.4 Assignment Decision Model

```python
@dataclass
class AssignmentDecision:
    vehicle_id: str
    bay_id: str
    confidence: float
    
    # Scoring breakdown
    proximity_score: float      # Distance to destination
    accessibility_score: float  # Accessibility match
    size_match_score: float     # Bay size vs vehicle size
    turnover_score: float       # Expected vs typical stay
    congestion_score: float     # Current traffic on that level
    priority_score: float       # Emergency/priority handling
    
    total_score: float
    reasoning: str
```

---

## 5. Neural Network Design

### 5.1 Network Architecture

```
INPUT LAYER (Feature Vector)
├── Vehicle Features (15 neurons)
│   ├── size_encoded [3]          # One-hot: compact/standard/large
│   ├── is_ev [1]
│   ├── has_blue_badge [1]
│   ├── has_children [1]
│   ├── mobility_level [3]        # One-hot
│   ├── purpose_encoded [12]      # One-hot: visit purposes
│   ├── estimated_stay_mins [1]   # Normalized
│   └── priority_level [1]        # Normalized
│
├── Destination Features (10 neurons)
│   ├── target_floor [1]          # Normalized -1 to 4
│   ├── shop_category [8]         # One-hot
│   └── num_destinations [1]      # Normalized
│
├── Temporal Features (8 neurons)
│   ├── hour_of_day [1]           # Normalized 0-1
│   ├── day_of_week [7]           # One-hot
│
├── Car Park State (20 neurons)
│   ├── level_occupancy [6]       # One per level
│   ├── level_congestion [6]      # Traffic flow score
│   ├── available_standard [1]
│   ├── available_blue_badge [1]
│   ├── available_parent_child [1]
│   ├── available_ev [1]
│   ├── queue_length [1]
│   └── recent_flow_rate [3]      # Entry/exit/internal
│
└── Bay Candidate Features (per bay, 12 neurons)
    ├── bay_type_encoded [4]
    ├── distance_to_destination [1]
    ├── distance_to_lift [1]
    ├── distance_to_exit [1]
    ├── level_normalized [1]
    ├── row_position [1]
    ├── current_neighbors_occupied [1]
    └── historical_turnover [1]

TOTAL INPUT: ~65 neurons + (12 × num_candidate_bays)
```

### 5.2 Hidden Layers

```
HIDDEN LAYER 1: 128 neurons (ReLU)
    └── Batch Normalization
    └── Dropout (0.2)

HIDDEN LAYER 2: 64 neurons (ReLU)
    └── Batch Normalization
    └── Dropout (0.2)

HIDDEN LAYER 3: 32 neurons (ReLU)
    └── Batch Normalization

OUTPUT LAYER: num_available_bays neurons (Softmax)
    └── Bay selection probabilities
```

### 5.3 Alternative: Attention-Based Architecture

For better handling of variable numbers of available bays:

```
Vehicle/Context Encoder (Transformer-style)
    └── Self-attention over input features
    └── Context vector (64 dims)

Bay Encoder (per candidate bay)
    └── MLP: 12 → 32 → 32
    └── Bay embedding (32 dims)

Cross-Attention
    └── Query: Context vector
    └── Key/Value: Bay embeddings
    └── Attention weights = Bay scores

Output
    └── Softmax over attention weights
    └── Selected bay = argmax
```

### 5.4 Network Implementation (PyTorch)

```python
class ParkingAssignmentNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Feature encoders
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(38, 64),  # destination + temporal + state
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        self.bay_encoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Combination layers
        self.combiner = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Scoring head
        self.scorer = nn.Linear(32 + 32, 1)  # Combined + bay = score
        
    def forward(self, vehicle_features, context_features, bay_features):
        # Encode inputs
        vehicle_emb = self.vehicle_encoder(vehicle_features)
        context_emb = self.context_encoder(context_features)
        
        # Combine vehicle and context
        combined = torch.cat([vehicle_emb, context_emb], dim=-1)
        combined = self.combiner(combined)
        
        # Score each bay
        bay_embs = self.bay_encoder(bay_features)  # [batch, num_bays, 32]
        
        # Expand combined for each bay
        combined_expanded = combined.unsqueeze(1).expand(-1, bay_embs.size(1), -1)
        
        # Concatenate and score
        bay_scores = self.scorer(
            torch.cat([combined_expanded, bay_embs], dim=-1)
        ).squeeze(-1)
        
        return F.softmax(bay_scores, dim=-1)
```

---

## 6. Genetic Algorithm Training

### 6.1 Chromosome Representation

Each individual in the population represents a complete neural network's weights:

```python
@dataclass
class Individual:
    id: str
    generation: int
    weights: np.ndarray       # Flattened network weights
    fitness: float = 0.0
    
    # Metadata
    parent_ids: Tuple[str, str] = None
    mutation_rate_used: float = 0.0
    
    def to_network(self) -> ParkingAssignmentNetwork:
        """Convert weights back to neural network"""
        pass
    
    def from_network(self, network: ParkingAssignmentNetwork):
        """Extract weights from network"""
        pass
```

### 6.2 Fitness Function

```python
def calculate_fitness(individual: Individual, 
                      simulation: Simulation,
                      num_episodes: int = 100) -> float:
    """
    Run simulation episodes and calculate fitness score.
    Higher is better.
    """
    network = individual.to_network()
    total_score = 0.0
    
    for episode in range(num_episodes):
        simulation.reset()
        episode_metrics = simulation.run(network, duration_hours=8)
        
        # Calculate weighted fitness components
        score = (
            # Primary metrics (higher weight)
            0.25 * episode_metrics.avg_proximity_score +
            0.20 * episode_metrics.accessibility_compliance +
            0.15 * episode_metrics.turnover_efficiency +
            
            # Secondary metrics
            0.10 * episode_metrics.congestion_reduction +
            0.10 * episode_metrics.prediction_accuracy +
            0.10 * (1.0 - episode_metrics.reassignment_rate) +
            
            # Penalties
            -0.05 * episode_metrics.invalid_assignments +
            -0.05 * episode_metrics.emergency_response_delay
        )
        
        total_score += max(0, score)
    
    return total_score / num_episodes

@dataclass
class EpisodeMetrics:
    avg_proximity_score: float      # 0-1, how close to destination
    accessibility_compliance: float # % correct bay type assignments
    turnover_efficiency: float      # Actual vs optimal throughput
    congestion_reduction: float     # vs baseline random assignment
    prediction_accuracy: float      # Predicted vs actual stay time
    reassignment_rate: float        # % cars that needed to move
    invalid_assignments: float      # Wrong size, type violations
    emergency_response_delay: float # Avg seconds for emergency assignment
```

### 6.3 Genetic Operators

```python
class GeneticTrainer:
    def __init__(self, config: GeneticConfig):
        self.population_size = config.population_size  # e.g., 100
        self.elite_count = config.elite_count          # e.g., 10
        self.mutation_rate = config.mutation_rate      # e.g., 0.1
        self.crossover_rate = config.crossover_rate    # e.g., 0.7
        
    def selection(self, population: List[Individual]) -> List[Individual]:
        """Tournament selection with elitism"""
        # Keep elite individuals
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        selected = sorted_pop[:self.elite_count]
        
        # Tournament selection for rest
        while len(selected) < self.population_size:
            tournament = random.sample(population, k=5)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover of weights"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        mask = np.random.random(parent1.weights.shape) > 0.5
        
        child1_weights = np.where(mask, parent1.weights, parent2.weights)
        child2_weights = np.where(mask, parent2.weights, parent1.weights)
        
        return (
            Individual(weights=child1_weights),
            Individual(weights=child2_weights)
        )
    
    def mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation of weights"""
        mutation_mask = np.random.random(individual.weights.shape) < self.mutation_rate
        noise = np.random.normal(0, 0.1, individual.weights.shape)
        
        individual.weights = np.where(
            mutation_mask,
            individual.weights + noise,
            individual.weights
        )
        return individual
    
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Create next generation"""
        # Selection
        selected = self.selection(population)
        
        # Crossover
        next_gen = selected[:self.elite_count]  # Elites pass through
        
        while len(next_gen) < self.population_size:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = self.crossover(p1, p2)
            next_gen.extend([c1, c2])
        
        # Mutation (except elites)
        for i in range(self.elite_count, len(next_gen)):
            next_gen[i] = self.mutate(next_gen[i])
        
        return next_gen[:self.population_size]
```

### 6.4 Training Loop

```python
def train_genetic_nn(
    car_park_config: CarParkConfig,
    generations: int = 500,
    population_size: int = 100,
    checkpoint_interval: int = 50
):
    """Main training loop"""
    
    # Initialize
    simulation = Simulation(car_park_config)
    trainer = GeneticTrainer(GeneticConfig(
        population_size=population_size,
        elite_count=10,
        mutation_rate=0.1,
        crossover_rate=0.7
    ))
    
    # Initial population (random weights)
    population = [
        Individual(weights=np.random.randn(TOTAL_WEIGHTS) * 0.1)
        for _ in range(population_size)
    ]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(generations):
        # Evaluate fitness (parallelizable)
        for individual in population:
            individual.fitness = calculate_fitness(individual, simulation)
        
        # Track progress
        fitnesses = [ind.fitness for ind in population]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        print(f"Gen {gen}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
        
        # Checkpoint
        if gen % checkpoint_interval == 0:
            best_individual = max(population, key=lambda x: x.fitness)
            save_checkpoint(best_individual, gen)
        
        # Evolve
        population = trainer.evolve_generation(population)
    
    # Return best
    return max(population, key=lambda x: x.fitness)
```

---

## 7. Simulation Engine

### 7.1 Core Simulation Loop

```python
class Simulation:
    def __init__(self, car_park: CarPark, config: SimConfig):
        self.car_park = car_park
        self.config = config
        self.clock = SimulationClock()
        self.vehicle_queue: Queue[Vehicle] = Queue()
        self.active_vehicles: Dict[str, Vehicle] = {}
        self.event_queue: PriorityQueue[Event] = PriorityQueue()
        
    def reset(self):
        """Reset simulation state"""
        self.clock.reset()
        self.vehicle_queue.clear()
        self.active_vehicles.clear()
        self.event_queue.clear()
        self.car_park.reset_all_bays()
        
    def run(self, network: ParkingAssignmentNetwork, 
            duration_hours: float) -> EpisodeMetrics:
        """Run single simulation episode"""
        
        metrics = MetricsCollector()
        end_time = self.clock.current + timedelta(hours=duration_hours)
        
        # Schedule initial arrival events
        self._schedule_arrivals(end_time)
        
        while self.clock.current < end_time:
            # Process next event
            event = self.event_queue.get()
            self.clock.advance_to(event.time)
            
            if event.type == EventType.VEHICLE_ARRIVAL:
                self._handle_arrival(event.vehicle, network, metrics)
                
            elif event.type == EventType.VEHICLE_DEPARTURE:
                self._handle_departure(event.vehicle, metrics)
                
            elif event.type == EventType.EMERGENCY:
                self._handle_emergency(event.vehicle, network, metrics)
        
        return metrics.compile()
    
    def _handle_arrival(self, vehicle: Vehicle, 
                        network: ParkingAssignmentNetwork,
                        metrics: MetricsCollector):
        """Process vehicle arrival and assign bay"""
        
        # Get available bays
        available_bays = self.car_park.get_available_bays(
            bay_type=self._required_bay_type(vehicle),
            min_size=vehicle.size
        )
        
        if not available_bays:
            metrics.record_rejection(vehicle)
            return
        
        # Encode features
        features = self._encode_features(vehicle, available_bays)
        
        # Get network prediction
        with torch.no_grad():
            bay_scores = network(*features)
        
        # Select bay
        selected_idx = bay_scores.argmax().item()
        selected_bay = available_bays[selected_idx]
        
        # Assign
        self._assign_bay(vehicle, selected_bay)
        
        # Predict stay duration
        vehicle.predicted_stay = self._predict_stay(vehicle)
        
        # Schedule departure
        actual_stay = self._sample_actual_stay(vehicle)
        departure_time = self.clock.current + actual_stay
        
        self.event_queue.put(Event(
            type=EventType.VEHICLE_DEPARTURE,
            time=departure_time,
            vehicle=vehicle
        ))
        
        # Record metrics
        metrics.record_assignment(vehicle, selected_bay, bay_scores)
```

### 7.2 Vehicle Generation

```python
class VehicleGenerator:
    """Generate realistic vehicle arrivals based on patterns"""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.purpose_distributions = self._load_purpose_distributions()
        self.arrival_patterns = self._load_arrival_patterns()
        
    def generate_arrival(self, current_time: datetime) -> Vehicle:
        """Generate a vehicle with realistic properties"""
        
        # Determine purpose based on time and day
        hour = current_time.hour
        day = current_time.weekday()
        purpose = self._sample_purpose(hour, day)
        
        # Vehicle properties based on purpose
        size = self._sample_size(purpose)
        has_blue_badge = random.random() < self.config.blue_badge_rate
        has_children = random.random() < self.config.parent_child_rate
        
        # Destination based on purpose
        destinations = self._sample_destinations(purpose)
        
        # Estimated stay
        estimated_stay = self._sample_estimated_stay(purpose)
        
        return Vehicle(
            id=str(uuid.uuid4()),
            arrival_time=current_time,
            size=size,
            is_ev=random.random() < self.config.ev_rate,
            has_blue_badge=has_blue_badge,
            has_children=has_children,
            mobility_requirements=self._sample_mobility(has_blue_badge),
            purpose=purpose,
            destination_shops=destinations,
            estimated_stay=estimated_stay,
            is_emergency=False,
            priority_level=0
        )
    
    def _sample_purpose(self, hour: int, day: int) -> VisitPurpose:
        """Sample purpose weighted by time/day patterns"""
        
        # Weekday patterns
        if day < 5:  # Mon-Fri
            if 7 <= hour < 10:
                # Morning: employees, quick shopping
                weights = {
                    VisitPurpose.WORK_EMPLOYEE: 0.3,
                    VisitPurpose.SHOPPING_QUICK: 0.2,
                    VisitPurpose.GROCERY: 0.3,
                    VisitPurpose.APPOINTMENT: 0.2
                }
            elif 12 <= hour < 14:
                # Lunch: dining, quick shopping
                weights = {
                    VisitPurpose.DINING: 0.4,
                    VisitPurpose.SHOPPING_QUICK: 0.3,
                    VisitPurpose.SHOPPING_REGULAR: 0.3
                }
            elif 17 <= hour < 20:
                # Evening: regular shopping, dining
                weights = {
                    VisitPurpose.SHOPPING_REGULAR: 0.4,
                    VisitPurpose.DINING: 0.3,
                    VisitPurpose.GROCERY: 0.2,
                    VisitPurpose.CINEMA: 0.1
                }
            else:
                weights = self._default_weights()
        else:
            # Weekend patterns
            weights = {
                VisitPurpose.SHOPPING_EXTENDED: 0.3,
                VisitPurpose.SHOPPING_REGULAR: 0.25,
                VisitPurpose.DINING: 0.2,
                VisitPurpose.CINEMA: 0.15,
                VisitPurpose.GROCERY: 0.1
            }
        
        return random.choices(
            list(weights.keys()),
            weights=list(weights.values())
        )[0]
```

### 7.3 Metrics Collection

```python
@dataclass
class MetricsCollector:
    assignments: List[AssignmentRecord] = field(default_factory=list)
    rejections: List[Vehicle] = field(default_factory=list)
    departures: List[DepartureRecord] = field(default_factory=list)
    emergency_responses: List[EmergencyRecord] = field(default_factory=list)
    
    def record_assignment(self, vehicle: Vehicle, bay: ParkingBay, 
                          scores: torch.Tensor):
        self.assignments.append(AssignmentRecord(
            vehicle_id=vehicle.id,
            bay_id=bay.id,
            bay_type=bay.bay_type,
            vehicle_requires=self._required_type(vehicle),
            distance_to_destination=self._calc_distance(vehicle, bay),
            time=datetime.now()
        ))
    
    def compile(self) -> EpisodeMetrics:
        """Compile collected data into metrics"""
        
        # Proximity score
        distances = [a.distance_to_destination for a in self.assignments]
        max_possible = 100  # meters
        avg_proximity = 1.0 - (sum(distances) / len(distances) / max_possible)
        
        # Accessibility compliance
        correct_type = sum(
            1 for a in self.assignments 
            if a.bay_type == a.vehicle_requires
        )
        accessibility_compliance = correct_type / len(self.assignments)
        
        # Prediction accuracy
        prediction_errors = [
            abs((d.actual_stay - d.predicted_stay).total_seconds())
            for d in self.departures
        ]
        avg_error_mins = sum(prediction_errors) / len(prediction_errors) / 60
        prediction_accuracy = max(0, 1.0 - avg_error_mins / 60)  # Within 60 mins = 0
        
        # ... calculate other metrics
        
        return EpisodeMetrics(
            avg_proximity_score=avg_proximity,
            accessibility_compliance=accessibility_compliance,
            prediction_accuracy=prediction_accuracy,
            # ...
        )
```

---

## 8. Streamlit UI Design

### 8.1 Page Structure

```
📁 streamlit_app/
├── 🏠 app.py                    # Main entry point
├── 📄 pages/
│   ├── 1_🚗_Live_Simulation.py
│   ├── 2_🧬_Training_Dashboard.py
│   ├── 3_📊_Analytics.py
│   ├── 4_⚙️_Configuration.py
│   └── 5_🗺️_Car_Park_Editor.py
├── 📦 components/
│   ├── car_park_visualizer.py
│   ├── metrics_dashboard.py
│   ├── training_charts.py
│   └── vehicle_inspector.py
└── 📦 utils/
    ├── state_manager.py
    └── data_export.py
```

### 8.2 Live Simulation Page

```python
# pages/1_🚗_Live_Simulation.py

import streamlit as st
from simulation import Simulation
from visualization import CarParkVisualizer

st.set_page_config(page_title="Live Simulation", layout="wide")

# Sidebar controls
st.sidebar.header("Simulation Controls")

speed = st.sidebar.slider("Simulation Speed", 1, 100, 10)
show_paths = st.sidebar.checkbox("Show Vehicle Paths", True)
show_heatmap = st.sidebar.checkbox("Show Occupancy Heatmap", False)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("▶️ Start"):
        st.session_state.running = True
with col2:
    if st.button("⏸️ Pause"):
        st.session_state.running = False
with col3:
    if st.button("🔄 Reset"):
        st.session_state.simulation.reset()

# Main content
st.title("🚗 Live Car Park Simulation")

# Two-column layout
left_col, right_col = st.columns([2, 1])

with left_col:
    # Car park visualization
    st.subheader("Car Park View")
    
    level_tabs = st.tabs(["Level -1", "Level 0", "Level 1", "Level 2", "Level 3", "Level 4"])
    
    for i, tab in enumerate(level_tabs):
        with tab:
            level_num = i - 1  # -1 to 4
            fig = CarParkVisualizer.render_level(
                st.session_state.simulation.car_park,
                level_num,
                show_paths=show_paths,
                show_heatmap=show_heatmap
            )
            st.plotly_chart(fig, use_container_width=True)

with right_col:
    # Real-time metrics
    st.subheader("📊 Live Metrics")
    
    metrics = st.session_state.simulation.get_current_metrics()
    
    st.metric("Total Occupancy", f"{metrics.occupancy_pct:.1f}%", 
              delta=f"{metrics.occupancy_change:+.1f}%")
    
    st.metric("Avg Wait Time", f"{metrics.avg_wait:.0f}s",
              delta=f"{metrics.wait_change:+.0f}s")
    
    st.metric("Assignment Accuracy", f"{metrics.accuracy:.1f}%")
    
    # Level breakdown
    st.subheader("Level Occupancy")
    for level in st.session_state.simulation.car_park.levels:
        occ = level.get_occupancy()
        st.progress(occ, text=f"Level {level.level_number}: {occ*100:.0f}%")
    
    # Recent arrivals
    st.subheader("Recent Activity")
    for event in st.session_state.simulation.recent_events[-5:]:
        st.text(f"{event.time.strftime('%H:%M')} - {event.description}")

# Vehicle queue
st.subheader("🚙 Incoming Queue")
queue_df = st.session_state.simulation.get_queue_dataframe()
st.dataframe(queue_df, use_container_width=True)
```

### 8.3 Training Dashboard Page

```python
# pages/2_🧬_Training_Dashboard.py

import streamlit as st
import plotly.graph_objects as go
from training import GeneticTrainer

st.set_page_config(page_title="Training Dashboard", layout="wide")

st.title("🧬 Genetic Neural Network Training")

# Training configuration
with st.expander("⚙️ Training Configuration", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        population_size = st.number_input("Population Size", 50, 500, 100)
        generations = st.number_input("Generations", 100, 2000, 500)
    
    with col2:
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1)
        crossover_rate = st.slider("Crossover Rate", 0.5, 1.0, 0.7)
    
    with col3:
        elite_count = st.number_input("Elite Count", 5, 50, 10)
        episodes_per_eval = st.number_input("Episodes per Evaluation", 10, 200, 50)

# Training controls
col1, col2, col3, col4 = st.columns(4)
with col1:
    start_training = st.button("🚀 Start Training", type="primary")
with col2:
    pause_training = st.button("⏸️ Pause")
with col3:
    resume_training = st.button("▶️ Resume")
with col4:
    stop_training = st.button("⏹️ Stop", type="secondary")

# Progress
if st.session_state.get('training_active', False):
    progress = st.session_state.trainer.progress
    st.progress(progress.generation / generations, 
                text=f"Generation {progress.generation}/{generations}")

# Fitness charts
st.subheader("📈 Fitness Evolution")

col1, col2 = st.columns(2)

with col1:
    # Best/Average fitness over generations
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.best_fitness_history,
        mode='lines',
        name='Best Fitness',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        y=st.session_state.avg_fitness_history,
        mode='lines',
        name='Average Fitness',
        line=dict(color='blue', width=1)
    ))
    fig.update_layout(title="Fitness Over Generations",
                      xaxis_title="Generation",
                      yaxis_title="Fitness Score")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Fitness distribution of current population
    fig = go.Figure(data=[go.Histogram(
        x=[ind.fitness for ind in st.session_state.population],
        nbinsx=30
    )])
    fig.update_layout(title="Population Fitness Distribution",
                      xaxis_title="Fitness",
                      yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# Component metrics
st.subheader("📊 Fitness Component Breakdown")

component_cols = st.columns(4)
components = ['proximity', 'accessibility', 'turnover', 'congestion']
for col, comp in zip(component_cols, components):
    with col:
        history = st.session_state.component_histories[comp]
        st.line_chart(history, height=150)
        st.caption(f"{comp.title()} Score")

# Best individual details
st.subheader("🏆 Best Individual")

best = max(st.session_state.population, key=lambda x: x.fitness)
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Fitness Score", f"{best.fitness:.4f}")
    st.metric("Generation", best.generation)
    st.text(f"ID: {best.id[:8]}...")
    
    if st.button("💾 Export Best Model"):
        # Export logic
        pass

with col2:
    # Network weight visualization
    st.text("Weight Distribution by Layer")
    # Histogram of weights per layer
```

### 8.4 Car Park Visualizer Component

```python
# components/car_park_visualizer.py

import plotly.graph_objects as go
import numpy as np

class CarParkVisualizer:
    
    COLORS = {
        'AVAILABLE': '#90EE90',      # Light green
        'OCCUPIED': '#FFB6C1',       # Light red
        'RESERVED': '#FFD700',       # Gold
        'BLUE_BADGE': '#4169E1',     # Royal blue
        'PARENT_CHILD': '#DDA0DD',   # Plum
        'EV': '#32CD32',             # Lime green
        'AISLE': '#E0E0E0',          # Light grey
        'LIFT': '#9370DB',           # Medium purple
        'RAMP': '#98FB98',           # Pale green
    }
    
    @classmethod
    def render_level(cls, car_park, level_num, 
                     show_paths=False, show_heatmap=False):
        """Render a single level of the car park"""
        
        level = car_park.get_level(level_num)
        
        fig = go.Figure()
        
        # Draw floor boundary
        fig.add_shape(
            type="rect",
            x0=0, y0=0,
            x1=level.width, y1=level.height,
            line=dict(color="black", width=2),
            fillcolor="#F5F5F5"
        )
        
        # Draw aisles
        for aisle in level.aisles:
            fig.add_shape(
                type="rect",
                x0=aisle.x, y0=aisle.y,
                x1=aisle.x + aisle.width,
                y1=aisle.y + aisle.height,
                fillcolor=cls.COLORS['AISLE'],
                line=dict(width=0)
            )
        
        # Draw parking bays
        for bay in level.bays:
            color = cls._get_bay_color(bay)
            
            fig.add_shape(
                type="rect",
                x0=bay.x, y0=bay.y,
                x1=bay.x + bay.width,
                y1=bay.y + bay.height,
                fillcolor=color,
                line=dict(color="gray", width=1)
            )
            
            # Bay label
            fig.add_annotation(
                x=bay.x + bay.width/2,
                y=bay.y + bay.height/2,
                text=bay.id,
                showarrow=False,
                font=dict(size=8)
            )
        
        # Draw facilities (lifts, stairs, etc.)
        for facility in level.facilities:
            cls._draw_facility(fig, facility)
        
        # Draw vehicle paths if enabled
        if show_paths:
            for vehicle in level.active_vehicles:
                if vehicle.path:
                    cls._draw_path(fig, vehicle.path)
        
        # Heatmap overlay
        if show_heatmap:
            cls._add_heatmap(fig, level)
        
        fig.update_layout(
            title=f"Level {level_num}: {level.name}",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False,
                      scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        
        return fig
    
    @classmethod
    def _get_bay_color(cls, bay):
        """Determine bay color based on type and status"""
        if bay.status == BayStatus.OCCUPIED:
            return cls.COLORS['OCCUPIED']
        elif bay.status == BayStatus.RESERVED:
            return cls.COLORS['RESERVED']
        elif bay.bay_type == BayType.BLUE_BADGE:
            return cls.COLORS['BLUE_BADGE']
        elif bay.bay_type == BayType.PARENT_CHILD:
            return cls.COLORS['PARENT_CHILD']
        elif bay.bay_type == BayType.EV:
            return cls.COLORS['EV']
        else:
            return cls.COLORS['AVAILABLE']
```

---

## 9. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 | Project setup | Repository, dependencies, CI/CD |
| 1.2 | Data models | Python classes for all entities |
| 1.3 | Car park config | JSON schema for Grand Arcade |
| 1.4 | Basic Streamlit shell | Navigation, page structure |

**Milestone:** Can load and display car park configuration

### Phase 2: Simulation Core (Weeks 3-4)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 | Event system | Event queue, clock, handlers |
| 2.2 | Vehicle generator | Realistic arrival patterns |
| 2.3 | Basic assignment | Random/greedy baseline |
| 2.4 | Metrics collection | All KPIs tracked |

**Milestone:** Simulation runs end-to-end with baseline algorithm

### Phase 3: Neural Network (Weeks 5-6)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1 | Feature engineering | All encoders implemented |
| 3.2 | Network architecture | PyTorch model |
| 3.3 | Inference pipeline | Fast bay selection |
| 3.4 | Model serialization | Save/load checkpoints |

**Milestone:** Neural network makes predictions (untrained)

### Phase 4: Genetic Algorithm (Weeks 7-8)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.1 | Population management | Creation, storage |
| 4.2 | Fitness evaluation | Parallel evaluation |
| 4.3 | Genetic operators | Selection, crossover, mutation |
| 4.4 | Training loop | Complete GA training |

**Milestone:** GA trains and improves network performance

### Phase 5: Visualization & UI (Weeks 9-10)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 5.1 | Car park visualizer | Interactive Plotly component |
| 5.2 | Live simulation page | Real-time updates |
| 5.3 | Training dashboard | Progress, charts |
| 5.4 | Analytics page | Historical analysis |

**Milestone:** Full Streamlit application functional

### Phase 6: Optimization & Polish (Weeks 11-12)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 6.1 | Performance tuning | Faster simulation |
| 6.2 | Model optimization | Better convergence |
| 6.3 | Edge cases | Emergency handling |
| 6.4 | Documentation | User guide, API docs |

**Milestone:** Production-ready system

---

## 10. Technical Requirements

### 10.1 Python Dependencies

```
# Core
python>=3.10
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0

# Machine Learning
torch>=2.0.0
scikit-learn>=1.3.0

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0

# Data & Config
pydantic>=2.0.0
pyyaml>=6.0

# Utilities
tqdm>=4.65.0
loguru>=0.7.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

### 10.2 Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | None | CUDA-capable |
| Storage | 10 GB | 50 GB SSD |

### 10.3 Project Structure

```
wpark-nn-integration/
├── 📄 README.md
├── 📄 WPARK_NN_SIMULATION_PLAN.md
├── 📄 requirements.txt
├── 📄 pyproject.toml
├── 📁 config/
│   ├── grand_arcade.json
│   ├── training_config.yaml
│   └── simulation_config.yaml
├── 📁 src/
│   ├── 📁 models/
│   │   ├── car_park.py
│   │   ├── vehicle.py
│   │   ├── bay.py
│   │   └── shop.py
│   ├── 📁 simulation/
│   │   ├── engine.py
│   │   ├── events.py
│   │   ├── vehicle_generator.py
│   │   └── metrics.py
│   ├── 📁 neural_network/
│   │   ├── network.py
│   │   ├── features.py
│   │   └── inference.py
│   ├── 📁 training/
│   │   ├── genetic.py
│   │   ├── fitness.py
│   │   └── population.py
│   └── 📁 utils/
│       ├── config.py
│       └── logging.py
├── 📁 streamlit_app/
│   ├── app.py
│   ├── 📁 pages/
│   └── 📁 components/
├── 📁 tests/
│   ├── test_simulation.py
│   ├── test_network.py
│   └── test_training.py
├── 📁 data/
│   ├── 📁 checkpoints/
│   └── 📁 logs/
└── 📁 docs/
    ├── architecture.md
    └── user_guide.md
```

---

## 11. Success Metrics

### 11.1 Training Metrics

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Fitness score | 0.3 | 0.7 | 0.85 |
| Convergence generations | N/A | <300 | <150 |
| Training time (500 gen) | N/A | <4 hours | <2 hours |

### 11.2 Assignment Quality Metrics

| Metric | Baseline (Random) | Target | Measurement |
|--------|-------------------|--------|-------------|
| Avg distance to destination | 45m | <25m | Per assignment |
| Accessibility compliance | 60% | >95% | Correct bay types |
| Stay prediction accuracy | ±45 min | ±15 min | 80th percentile |
| Emergency response time | 60s | <15s | Time to assign |

### 11.3 Operational Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Throughput (cars/hour) | 80 | 100+ |
| Avg queue wait time | 120s | <60s |
| Bay utilization efficiency | 65% | >80% |
| Level load balancing | σ=15% | σ<8% |

---

## 12. Risk Assessment

### 12.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GA doesn't converge | High | Medium | Tune hyperparameters, larger population |
| Simulation too slow | Medium | Medium | Profile, optimize, parallelize |
| Overfitting to patterns | High | Low | Diverse training scenarios |
| Memory issues | Medium | Low | Batch processing, cleanup |

### 12.2 Project Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Scope creep | Medium | High | Strict phase gates |
| Unrealistic targets | Medium | Medium | Regular metric review |
| Dependency issues | Low | Medium | Pin versions, test early |

---

## Appendix A: Grand Arcade Specific Configuration

```json
{
  "car_park": {
    "id": "grand-arcade-cambridge",
    "name": "Grand Arcade Car Park",
    "location": "Cambridge, UK",
    "total_capacity": 951,
    "height_limit_m": 1.98,
    "levels": [
      {
        "level_number": -1,
        "name": "Underground",
        "capacity": 160,
        "blue_badge_bays": 8,
        "parent_child_bays": 6,
        "connected_mall_floor": "Lower Ground",
        "shops": ["John Lewis Basement", "Waitrose"],
        "facilities": ["Shopmobility", "Changing Places WC"]
      },
      {
        "level_number": 0,
        "name": "Entry/Exit",
        "capacity": 0,
        "is_entry_exit": true,
        "connected_mall_floor": "Ground",
        "shops": ["John Lewis Main", "H&M", "Zara", "Costa"]
      },
      {
        "level_number": 1,
        "name": "Level 1",
        "capacity": 200,
        "connected_mall_floor": "First Floor",
        "shops": ["Apple Store", "Superdry", "Pandora"]
      },
      {
        "level_number": 2,
        "name": "Level 2 (Accessible)",
        "capacity": 200,
        "blue_badge_bays": 26,
        "parent_child_bays": 6,
        "connected_mall_floor": "Second Floor",
        "shops": ["John Lewis Electricals", "Food Court"],
        "is_main_accessible": true
      },
      {
        "level_number": 3,
        "name": "Level 3",
        "capacity": 195
      },
      {
        "level_number": 4,
        "name": "Level 4 (Top)",
        "capacity": 196
      }
    ]
  }
}
```

---

## Appendix B: Sample Training Results Template

```
═══════════════════════════════════════════════════════════════════
                    TRAINING RUN SUMMARY
═══════════════════════════════════════════════════════════════════

Run ID:         run_20260407_143022
Duration:       3h 42m 18s
Generations:    500
Final Best:     0.8234

───────────────────────────────────────────────────────────────────
                    FITNESS PROGRESSION
───────────────────────────────────────────────────────────────────

Gen 0:    Best: 0.2341  Avg: 0.1823  Min: 0.0912
Gen 50:   Best: 0.4521  Avg: 0.3892  Min: 0.2341
Gen 100:  Best: 0.5834  Avg: 0.5102  Min: 0.3892
Gen 200:  Best: 0.6923  Avg: 0.6234  Min: 0.4892
Gen 300:  Best: 0.7534  Avg: 0.6892  Min: 0.5234
Gen 400:  Best: 0.7923  Avg: 0.7234  Min: 0.5892
Gen 500:  Best: 0.8234  Avg: 0.7523  Min: 0.6234

───────────────────────────────────────────────────────────────────
                    COMPONENT BREAKDOWN (Best Individual)
───────────────────────────────────────────────────────────────────

Proximity Score:          0.89
Accessibility Compliance: 0.97
Turnover Efficiency:      0.78
Congestion Reduction:     0.71
Prediction Accuracy:      0.82

───────────────────────────────────────────────────────────────────
                    MODEL SAVED
───────────────────────────────────────────────────────────────────

Checkpoint: data/checkpoints/best_model_20260407.pt
Weights:    data/checkpoints/best_weights_20260407.npy

═══════════════════════════════════════════════════════════════════
```

---

*Document prepared for WPARK Neural Network Integration Project*
