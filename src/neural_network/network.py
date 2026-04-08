"""Neural network for parking bay assignment using NumPy."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

from ..models.car_park import CarPark, ParkingBay, BayType, BayStatus
from ..models.vehicle import Vehicle, VehicleSize, VisitPurpose, MobilityLevel


@dataclass
class LayerInfo:
    """Information about a network layer for visualization."""
    name: str
    input_size: int
    output_size: int
    weights: np.ndarray
    biases: np.ndarray
    activations: Optional[np.ndarray] = None


class NeuralNetwork:
    """
    Feedforward neural network for parking assignment.
    
    Architecture:
    - Input: Vehicle features + Context features (53 neurons)
    - Hidden 1: 64 neurons (ReLU)
    - Hidden 2: 32 neurons (ReLU)
    - Output: Bay scores (variable, softmax)
    """
    
    # Input feature sizes
    VEHICLE_FEATURES = 15  # size(4) + ev(1) + blue_badge(1) + parent_child(1) + mobility(5) + priority(1) + stay(1) + purpose(7) - overlaps
    CONTEXT_FEATURES = 10  # level_occupancies(3) + queue(1) + time_features(6)
    BAY_FEATURES = 8       # type(4) + distance(1) + level(1) + occupancy_nearby(1) + size(1)
    
    def __init__(self, hidden_sizes: List[int] = None):
        """Initialize network with random weights."""
        self.hidden_sizes = hidden_sizes or [64, 32]
        
        # Total input for vehicle/context encoding
        self.input_size = self.VEHICLE_FEATURES + self.CONTEXT_FEATURES
        
        # Build layer sizes
        layer_sizes = [self.input_size] + self.hidden_sizes
        
        # Initialize weights with Xavier/Glorot initialization
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(np.random.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))
        
        # Bay scoring layer (from last hidden to single score per bay)
        # Takes combined features + bay features
        self.bay_scorer_weights = np.random.uniform(
            -0.1, 0.1, (self.hidden_sizes[-1] + self.BAY_FEATURES, 1)
        )
        self.bay_scorer_bias = np.zeros(1)
        
        # Store last activations for visualization
        self.last_activations: List[np.ndarray] = []
        self.last_input: Optional[np.ndarray] = None
        self.last_output: Optional[np.ndarray] = None
        self.last_bay_scores: Optional[np.ndarray] = None
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation with numerical stability."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x) + 1e-8)
    
    def encode_vehicle(self, vehicle: Vehicle) -> np.ndarray:
        """Encode vehicle into feature vector."""
        features = []
        
        # Size one-hot (4)
        size_idx = list(VehicleSize).index(vehicle.size)
        size_onehot = [0] * 4
        size_onehot[size_idx] = 1
        features.extend(size_onehot)
        
        # EV flag (1)
        features.append(1.0 if vehicle.is_ev else 0.0)
        
        # Accessibility flags (2)
        features.append(1.0 if vehicle.requires_blue_badge else 0.0)
        features.append(1.0 if vehicle.requires_parent_child else 0.0)
        
        # Mobility one-hot (5)
        mob_idx = list(MobilityLevel).index(vehicle.mobility)
        mob_onehot = [0] * 5
        mob_onehot[mob_idx] = 1
        features.extend(mob_onehot)
        
        # Priority (1) - normalized
        features.append(vehicle.priority)
        
        # Estimated stay (1) - normalized to 0-1 (assume max 8 hours)
        features.append(min(vehicle.estimated_stay_minutes / 480.0, 1.0))
        
        # Purpose encoding - simplified to importance score (1)
        purpose_scores = {
            VisitPurpose.EMERGENCY: 1.0,
            VisitPurpose.MEDICAL: 0.8,
            VisitPurpose.QUICK_ERRAND: 0.7,
            VisitPurpose.SHOPPING: 0.5,
            VisitPurpose.DINING: 0.5,
            VisitPurpose.ENTERTAINMENT: 0.4,
            VisitPurpose.COMMUTE: 0.3
        }
        features.append(purpose_scores.get(vehicle.purpose, 0.5))
        
        return np.array(features, dtype=np.float32)
    
    def encode_context(self, car_park: CarPark, current_time: float, queue_length: int) -> np.ndarray:
        """Encode car park context into feature vector."""
        features = []
        
        # Level occupancies (3 levels)
        for level in car_park.levels[:3]:
            features.append(level.occupancy_rate)
        
        # Pad if fewer levels
        while len(features) < 3:
            features.append(0.0)
        
        # Queue length (normalized, assume max 20)
        features.append(min(queue_length / 20.0, 1.0))
        
        # Time of day (sin/cos encoding for cyclical, 2)
        hour = (current_time / 60.0) % 24
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        # Day progress (1)
        features.append((current_time / 60.0) / 24.0 % 1.0)
        
        # Overall car park stats (3)
        features.append(car_park.overall_occupancy)
        features.append(car_park.total_available / max(car_park.total_capacity, 1))
        features.append(min(car_park.total_occupied / 100.0, 1.0))  # normalized count
        
        return np.array(features, dtype=np.float32)
    
    def encode_bay(self, bay: ParkingBay, target_floor: int) -> np.ndarray:
        """Encode a bay into feature vector."""
        features = []
        
        # Bay type one-hot (4)
        type_idx = list(BayType).index(bay.bay_type)
        type_onehot = [0] * 4
        type_onehot[type_idx] = 1
        features.extend(type_onehot)
        
        # Distance to lift (normalized, assume max 100)
        features.append(min(bay.distance_to_lift / 100.0, 1.0))
        
        # Level distance from target (normalized, max 3 floors)
        level_num = int(bay.level_id.split('_')[1]) if '_' in bay.level_id else 0
        features.append(abs(level_num - target_floor) / 3.0)
        
        # Position in row (normalized)
        features.append(bay.position / 20.0)
        
        # Bay size score
        size_scores = {'compact': 0.3, 'standard': 0.5, 'large': 0.8}
        features.append(size_scores.get(bay.size.value, 0.5))
        
        return np.array(features, dtype=np.float32)
    
    def forward(self, vehicle_features: np.ndarray, context_features: np.ndarray, 
                bay_features: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            vehicle_features: (VEHICLE_FEATURES,) vehicle encoding
            context_features: (CONTEXT_FEATURES,) context encoding
            bay_features: (num_bays, BAY_FEATURES) bay encodings
            
        Returns:
            Bay selection probabilities (num_bays,)
        """
        # Combine vehicle and context
        x = np.concatenate([vehicle_features, context_features])
        self.last_input = x.copy()
        self.last_activations = [x.copy()]
        
        # Pass through hidden layers
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = self.relu(x @ w + b)
            self.last_activations.append(x.copy())
        
        # Score each bay
        num_bays = bay_features.shape[0]
        scores = np.zeros(num_bays)
        
        for i in range(num_bays):
            # Combine context encoding with bay features
            combined = np.concatenate([x, bay_features[i]])
            scores[i] = (combined @ self.bay_scorer_weights + self.bay_scorer_bias)[0]
        
        self.last_bay_scores = scores.copy()
        
        # Softmax for probabilities
        probs = self.softmax(scores)
        self.last_output = probs.copy()
        
        return probs
    
    def predict(self, vehicle: Vehicle, car_park: CarPark, 
                available_bays: List[ParkingBay], current_time: float,
                queue_length: int = 0) -> Tuple[Optional[str], np.ndarray]:
        """
        Predict best bay for a vehicle.
        
        Returns:
            Tuple of (best_bay_id or None, probabilities array)
        """
        if not available_bays:
            return None, np.array([])
        
        # Encode inputs
        vehicle_features = self.encode_vehicle(vehicle)
        context_features = self.encode_context(car_park, current_time, queue_length)
        
        target_floor = vehicle.target_floor if vehicle.target_floor is not None else 0
        bay_features = np.array([self.encode_bay(bay, target_floor) for bay in available_bays])
        
        # Forward pass
        probs = self.forward(vehicle_features, context_features, bay_features)
        
        # Select highest probability bay
        best_idx = np.argmax(probs)
        return available_bays[best_idx].id, probs
    
    def get_weights_flat(self) -> np.ndarray:
        """Get all weights as a flat array for genetic algorithm."""
        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.extend(w.flatten())
            flat.extend(b.flatten())
        flat.extend(self.bay_scorer_weights.flatten())
        flat.extend(self.bay_scorer_bias.flatten())
        return np.array(flat, dtype=np.float32)
    
    def set_weights_flat(self, flat_weights: np.ndarray):
        """Set weights from a flat array."""
        idx = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_size = w.size
            b_size = b.size
            self.weights[i] = flat_weights[idx:idx + w_size].reshape(w.shape)
            idx += w_size
            self.biases[i] = flat_weights[idx:idx + b_size].reshape(b.shape)
            idx += b_size
        
        # Bay scorer
        w_size = self.bay_scorer_weights.size
        b_size = self.bay_scorer_bias.size
        self.bay_scorer_weights = flat_weights[idx:idx + w_size].reshape(self.bay_scorer_weights.shape)
        idx += w_size
        self.bay_scorer_bias = flat_weights[idx:idx + b_size].reshape(self.bay_scorer_bias.shape)
    
    def get_layer_info(self) -> List[LayerInfo]:
        """Get layer information for visualization."""
        layers = []
        
        # Input layer
        layers.append(LayerInfo(
            name="Input",
            input_size=0,
            output_size=self.input_size,
            weights=np.array([]),
            biases=np.array([]),
            activations=self.last_activations[0] if self.last_activations else None
        ))
        
        # Hidden layers
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            layers.append(LayerInfo(
                name=f"Hidden {i+1}",
                input_size=w.shape[0],
                output_size=w.shape[1],
                weights=w,
                biases=b,
                activations=self.last_activations[i + 1] if len(self.last_activations) > i + 1 else None
            ))
        
        # Output layer
        layers.append(LayerInfo(
            name="Bay Scorer",
            input_size=self.hidden_sizes[-1] + self.BAY_FEATURES,
            output_size=1,
            weights=self.bay_scorer_weights,
            biases=self.bay_scorer_bias,
            activations=self.last_output
        ))
        
        return layers
    
    def save(self, filepath: str):
        """Save network weights to file."""
        data = {
            'hidden_sizes': self.hidden_sizes,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'bay_scorer_weights': self.bay_scorer_weights.tolist(),
            'bay_scorer_bias': self.bay_scorer_bias.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetwork':
        """Load network weights from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        network = cls(hidden_sizes=data['hidden_sizes'])
        network.weights = [np.array(w) for w in data['weights']]
        network.biases = [np.array(b) for b in data['biases']]
        network.bay_scorer_weights = np.array(data['bay_scorer_weights'])
        network.bay_scorer_bias = np.array(data['bay_scorer_bias'])
        
        return network


class NeuralNetworkAssignment:
    """Assignment strategy using neural network."""
    
    def __init__(self, network: NeuralNetwork = None):
        self.network = network or NeuralNetwork()
        self.last_prediction_info: Dict = {}
    
    def assign(self, vehicle: Vehicle, car_park: CarPark, shops: List,
               shop_floor_map: Dict[str, int], current_time: float = 0.0,
               queue_length: int = 0) -> Optional[str]:
        """Assign a bay using neural network prediction."""
        # Get available bays
        available_bays = car_park.get_all_available_bays()
        
        if not available_bays:
            self.last_prediction_info = {'error': 'No available bays'}
            return None
        
        # Handle special requirements first
        required_type = None
        if vehicle.requires_blue_badge:
            required_type = BayType.BLUE_BADGE
        elif vehicle.requires_parent_child:
            required_type = BayType.PARENT_CHILD
        elif vehicle.requires_ev_charging:
            required_type = BayType.EV
        
        # Filter by required type if needed
        if required_type:
            type_bays = [b for b in available_bays if b.bay_type == required_type]
            if type_bays:
                available_bays = type_bays
        
        # Get prediction from network
        best_bay_id, probs = self.network.predict(
            vehicle, car_park, available_bays, current_time, queue_length
        )
        
        # Store prediction info for visualization
        if len(probs) > 0:
            sorted_probs = sorted(probs.tolist(), reverse=True)[:3]
        else:
            sorted_probs = []
            
        self.last_prediction_info = {
            'vehicle_id': vehicle.id,
            'available_count': len(available_bays),
            'selected_bay': best_bay_id,
            'confidence': float(np.max(probs)) if len(probs) > 0 else 0.0,
            'top_3_probs': sorted_probs
        }
        
        return best_bay_id
