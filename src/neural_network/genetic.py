"""Genetic algorithm trainer for neural network optimization."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import numpy as np
import random
import uuid

from .network import NeuralNetwork


@dataclass
class Individual:
    """An individual in the genetic population (represents one neural network)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    fitness: float = 0.0
    
    # Training metadata
    parent_ids: Optional[Tuple[str, str]] = None
    mutation_rate_used: float = 0.0
    episodes_evaluated: int = 0
    
    def to_network(self) -> NeuralNetwork:
        """Convert individual weights to neural network."""
        network = NeuralNetwork()
        if len(self.weights) > 0:
            network.set_weights_flat(self.weights)
        return network
    
    @classmethod
    def from_network(cls, network: NeuralNetwork, generation: int = 0) -> 'Individual':
        """Create individual from a neural network."""
        return cls(
            generation=generation,
            weights=network.get_weights_flat()
        )
    
    def copy(self) -> 'Individual':
        """Create a copy of this individual."""
        return Individual(
            id=str(uuid.uuid4())[:8],
            generation=self.generation,
            weights=self.weights.copy(),
            fitness=self.fitness,
            parent_ids=self.parent_ids,
            mutation_rate_used=self.mutation_rate_used
        )


@dataclass
class TrainingStats:
    """Statistics for genetic algorithm training."""
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    fitness_std: float = 0.0
    
    # History
    fitness_history: List[float] = field(default_factory=list)
    best_individual_id: str = ""


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    elite_count: int = 5
    mutation_rate: float = 0.15
    mutation_strength: float = 0.3   # was 0.1 — bigger perturbations needed to escape plateaus
    crossover_rate: float = 0.7
    tournament_size: int = 5

    # Adaptive mutation
    adaptive_mutation: bool = True
    mutation_rate_min: float = 0.05
    mutation_rate_max: float = 0.5

    # Diversity maintenance — inject fresh random individuals when population collapses
    diversity_threshold: float = 0.5   # reinject when avg pairwise distance falls below this
    diversity_inject_frac: float = 0.3  # replace this fraction of the worst individuals


class GeneticTrainer:
    """Genetic algorithm trainer for neural networks."""
    
    def __init__(self, config: GeneticConfig = None):
        self.config = config or GeneticConfig()
        self.population: List[Individual] = []
        self.generation: int = 0
        self.stats = TrainingStats()
        self.best_individual: Optional[Individual] = None
        
        # Callbacks for UI updates
        self.on_generation_complete: Optional[Callable[[TrainingStats], None]] = None
        self.on_fitness_evaluated: Optional[Callable[[Individual], None]] = None
    
    def initialize_population(self):
        """Initialize population with random individuals."""
        self.population = []
        for _ in range(self.config.population_size):
            network = NeuralNetwork()
            individual = Individual.from_network(network, generation=0)
            self.population.append(individual)
        self.generation = 0
    
    def selection(self) -> List[Individual]:
        """Tournament selection with elitism."""
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # Keep elite
        selected = [ind.copy() for ind in sorted_pop[:self.config.elite_count]]
        
        # Tournament selection for rest
        while len(selected) < self.config.population_size:
            tournament = random.sample(self.population, k=min(self.config.tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner.copy())
        
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover of weights."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover
        mask = np.random.random(parent1.weights.shape) > 0.5
        
        child1_weights = np.where(mask, parent1.weights, parent2.weights)
        child2_weights = np.where(mask, parent2.weights, parent1.weights)
        
        child1 = Individual(
            generation=self.generation + 1,
            weights=child1_weights,
            parent_ids=(parent1.id, parent2.id)
        )
        child2 = Individual(
            generation=self.generation + 1,
            weights=child2_weights,
            parent_ids=(parent1.id, parent2.id)
        )
        
        return child1, child2
    
    def mutate(self, individual: Individual, strength_scale: float = 1.0) -> Individual:
        """Gaussian mutation of weights.

        strength_scale > 1 is used during diversity-injection bursts so that
        fresh candidates explore more aggressively.
        """
        mutation_rate = self.config.mutation_rate

        if self.config.adaptive_mutation and self.best_individual:
            best_f = self.best_individual.fitness + 1e-8
            # Individuals far below the best get MORE mutation (exploration).
            # Individuals close to the best get LESS mutation (exploitation).
            fitness_ratio = min(individual.fitness / best_f, 1.0)
            mutation_rate = self.config.mutation_rate_min + (
                (self.config.mutation_rate_max - self.config.mutation_rate_min) * (1.0 - fitness_ratio)
            )
            mutation_rate = np.clip(mutation_rate, self.config.mutation_rate_min, self.config.mutation_rate_max)

        effective_strength = self.config.mutation_strength * strength_scale

        mutation_mask = np.random.random(individual.weights.shape) < mutation_rate
        noise = np.random.normal(0, effective_strength, individual.weights.shape)

        individual.weights = np.where(mutation_mask, individual.weights + noise, individual.weights)
        individual.mutation_rate_used = mutation_rate

        return individual
    
    def evolve_generation(self) -> List[Individual]:
        """Create the next generation."""
        # ── Diversity check ────────────────────────────────────────
        # If population has converged, replace the bottom fraction with fresh
        # random individuals so the search doesn't stall permanently.
        diversity = self.get_population_diversity()
        inject_fresh = (
            self.config.diversity_threshold > 0
            and diversity < self.config.diversity_threshold
            and len(self.population) > self.config.elite_count + 1
        )

        # Selection
        selected = self.selection()

        # Create new population
        new_population = []

        # Keep elite unchanged
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for ind in sorted_pop[:self.config.elite_count]:
            elite = ind.copy()
            elite.generation = self.generation + 1
            new_population.append(elite)

        # How many slots are reserved for injected fresh individuals
        n_inject = (
            max(1, int(self.config.population_size * self.config.diversity_inject_frac))
            if inject_fresh else 0
        )
        n_offspring = self.config.population_size - self.config.elite_count - n_inject

        # Crossover and mutation for regular offspring
        while len(new_population) < self.config.elite_count + n_offspring:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.append(child1)
            if len(new_population) < self.config.elite_count + n_offspring:
                new_population.append(child2)

        # Fresh random individuals to restore diversity
        for _ in range(n_inject):
            if len(new_population) >= self.config.population_size:
                break
            network = NeuralNetwork()
            fresh = Individual.from_network(network, generation=self.generation + 1)
            # Give fresh individuals a strong initial mutation so they explore broadly
            fresh = self.mutate(fresh, strength_scale=2.0)
            new_population.append(fresh)

        self.generation += 1
        self.population = new_population[:self.config.population_size]

        return self.population
    
    def update_stats(self):
        """Update training statistics."""
        fitnesses = [ind.fitness for ind in self.population]
        
        self.stats.generation = self.generation
        self.stats.best_fitness = max(fitnesses)
        self.stats.avg_fitness = np.mean(fitnesses)
        self.stats.worst_fitness = min(fitnesses)
        self.stats.fitness_std = np.std(fitnesses)
        self.stats.fitness_history.append(self.stats.best_fitness)
        
        # Track best individual
        best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or best.fitness > self.best_individual.fitness:
            self.best_individual = best.copy()
        self.stats.best_individual_id = self.best_individual.id
    
    def get_best_network(self) -> NeuralNetwork:
        """Get the best neural network from the population."""
        if self.best_individual:
            return self.best_individual.to_network()
        if self.population:
            best = max(self.population, key=lambda x: x.fitness)
            return best.to_network()
        return NeuralNetwork()
    
    def get_population_diversity(self) -> float:
        """Calculate population diversity (average pairwise distance)."""
        if len(self.population) < 2:
            return 0.0
        
        # Sample pairs for efficiency
        sample_size = min(20, len(self.population))
        sample = random.sample(self.population, sample_size)
        
        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                dist = np.linalg.norm(sample[i].weights - sample[j].weights)
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
