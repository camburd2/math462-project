"""A Boid (bird-oid) agent for implementing Craig Reynolds's Boids flocking model.

This implementation uses numpy arrays to represent vectors for efficient computation
of flocking behavior.
"""
import numpy as np
from utils import get
from mesa.experimental.continuous_space import ContinuousSpaceAgent


class Prey(ContinuousSpaceAgent):
    """A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents
        - Separation: avoiding getting too close to any other agent
        - Alignment: trying to fly in the same direction as neighbors

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and direction (a vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    """

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        cruise_speed=get('prey_cruise_speed'),
        burst_speed=get('prey_burst_speed'),
        direction=(1, 1),
        vision=get('vision'),
        separation=get('separation'),
        cohere=get('cohere'),
        separate=get('separate'),
        match=get('match'),
    ):
        """Create a new Boid flocker agent.

        Args:
            model: Model instance the agent belongs to
            speed: Distance to move per step
            direction: numpy vector for the Boid's direction of movement
            vision: Radius to look around for nearby Boids
            separation: Minimum distance to maintain from other Boids
            cohere: Relative importance of matching neighbors' positions (default: 0.03)
            separate: Relative importance of avoiding close neighbors (default: 0.015)
            match: Relative importance of matching neighbors' directions (default: 0.05)
        """
        super().__init__(space, model)
        self.position = np.array(position) # Ensure position is numpy array
        self.cruise_speed = cruise_speed
        self.burst_speed = burst_speed
        norm = np.linalg.norm(direction)
        self.direction = direction / norm if norm > 0 else np.array([1.0, 0.0])
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.neighbors = []
        self.predators_nearby = []

    def step(self):
        """Get neighbors, calculate flocking and move."""
        all_neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
        # Store all neighbors (excluding self) for potential future use
        self.neighbors = [n for n in all_neighbors if n is not self]

        # Separate neighbors into Prey (for flocking) and Predators (for escaping)
        flock_neighbors = []
        predator_neighbors = []
        for n in self.neighbors:
            if isinstance(n, Predator):
                predator_neighbors.append(n)
            else:
                flock_neighbors.append(n)
        self.predators_nearby = predator_neighbors # Store for optional use


        # --- Calculate Flocking Vectors (based on Prey neighbors) ---
        cohere_vector = np.zeros(2)
        separate_vector = np.zeros(2)
        match_vector = np.zeros(2)

        if flock_neighbors:
            delta_flock = self.space.calculate_difference_vector(self.position, agents=flock_neighbors)
            flock_distances = np.linalg.norm(delta_flock, axis=1)

            cohere_vector = delta_flock.sum(axis=0) * self.cohere_factor
            # Filter delta using flock_distances for separation
            separate_vector = (
                -1 * delta_flock[flock_distances < self.separation].sum(axis=0) * self.separate_factor
            )
            match_vector = (
                np.asarray([n.direction for n in flock_neighbors]).sum(axis=0) * self.match_factor
            )

        if predator_neighbors:
            # --- Flee directly away from predators ---
            # 1.  Build a single escape vector that points away from every predator
            escape_vec = np.zeros(2)
            for predator in predator_neighbors:
                away = self.position - predator.position      # vector pointing from predator to prey
                norm = np.linalg.norm(away)
                if norm > 0:
                    escape_vec += away / norm                 # add the unit vector

            # 2.  Normalise the combined escape vector
            norm = np.linalg.norm(escape_vec)
            if norm > 0:
                self.direction = escape_vec / norm

            # 3.  Move faster than normal to create distance
            new_position = self.position + self.direction * self.burst_speed
            self.position = new_position % self.space.size
        else:
            # --- Combine Vectors and Update Direction ---
            flocking_vector = cohere_vector + separate_vector + match_vector

            # Normalize the target direction vector IF it has magnitude
            norm = np.linalg.norm(flocking_vector)
            if norm > 0:
                # If there are influences, SET (=) direction towards the normalized target
                self.direction = flocking_vector / norm
            # else: If target_direction_vector is zero, agent keeps its previous direction.

            # --- Move Agent ---
            # (Movement code using self.position = ... % self.space.size remains the same)
            new_position = self.position + self.direction * self.cruise_speed
            self.position = new_position % self.space.size


class Predator(ContinuousSpaceAgent):
    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        cruise_speed=get('predator_cruise_speed'),
        burst_speed=get('predator_burst_speed'),
        direction=(1, 0),
        vision=get('predator_vision'),
        threshold=get('threshold'),
        eat_radius=get('eat_radius'),
        burst_dist=get('burst_dist'),
        burst_cooldown=get('predator_burst_cooldown'),
        burst_length=get('predator_burst_length'),
        eat_cooldown=get('predator_eat_cooldown'),
        max_turn_angle=None
    ):
        super().__init__(space, model)
        self.position = np.array(position) # Ensure position is numpy array
        self.cruise_speed = cruise_speed
        self.burst_speed = burst_speed
        # Normalize initial direction
        norm = np.linalg.norm(direction)
        self.direction = direction / norm if norm > 0 else np.array([1.0, 0.0])
        self.vision = vision
        self.burst_length = burst_length

        self.eaten_count = 0
        self.threshold = threshold
        self.eat_radius = eat_radius
        self.burst_dist = float(burst_dist)

        self.burst_cooldown = burst_cooldown
        self.eat_cooldown = eat_cooldown
        self.eat_cooldown_count = 0
        self.cooldown = 0
        self.burst_remaining = 0
        self.max_turn_angle = max_turn_angle

    def step(self):
        """Find the nearest Prey and move towards it."""
        if self.burst_remaining > 0:
            self.burst_remaining -= 1
            if self.burst_remaining == 0:
                self.cooldown = self.burst_cooldown
        elif self.cooldown > 0:
            self.cooldown -= 1

        if self.eat_cooldown_count > 0:
            self.eat_cooldown_count -= 1

        # Find all neighbors within vision radius
        neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)

        # Filter for Prey agents
        prey_neighbors = [
            (n, d) for n, d in zip(neighbors, distances) if isinstance(n, Prey)
        ]

        if prey_neighbors and (self.eaten_count < self.threshold):
            # Find the closest Prey
            closest_neighbor, dist_nearest = min(prey_neighbors, key=lambda item: item[1])
            
            # ----- DIRECTION -----  
            target_vector = closest_neighbor.position - self.position
            norm = np.linalg.norm(target_vector)
            if norm > 0:
                desired_dir = target_vector / norm
                # signed angle between curr dir and desired dir
                dot = np.clip(np.dot(self.direction, desired_dir), -1.0, 1.0)
                ang = np.arccos(dot)

                if ang > self.max_turn_angle:                 # clamp turn
                    sign = np.sign(np.cross(self.direction, desired_dir))
                    ca, sa = np.cos(self.max_turn_angle), np.sin(self.max_turn_angle) * sign
                    rot    = np.array([[ca, -sa], [sa, ca]])  # 2‑D rotation matrix
                    new_dir = rot @ self.direction
                    self.direction = new_dir / np.linalg.norm(new_dir)
                else:                                         # can turn fully
                    self.direction = desired_dir

            # ----- SPEED -----            
            if (
                (self.burst_remaining == 0)
                and (self.cooldown == 0)
                and (dist_nearest <= self.burst_dist)
                and (self.eat_cooldown_count == 0)
            ):
                self.burst_remaining = self.burst_length
            
            speed = self.burst_speed if self.burst_remaining > 0 else self.cruise_speed

            self.position = (self.position + self.direction * speed) % self.space.size

            # did we catch a prey? (within 2 unit)
            if (
                self.eat_cooldown_count == 0
                and np.linalg.norm(self.position - closest_neighbor.position) <= self.eat_radius
                and closest_neighbor not in self.model.prey_to_remove
            ):
                self.model.prey_to_remove.append(closest_neighbor)
                self.eaten_count += 1
                self.eat_cooldown_count = self.eat_cooldown
        else:
            # No Preys nearby: Optionally add random wander, or just continue
            # Let's add a small random perturbation to direction to avoid getting stuck
            random_turn_angle = self.model.random.uniform(-0.1, 0.1) # Small random turn
            cos_a, sin_a = np.cos(random_turn_angle), np.sin(random_turn_angle)
            self.direction = np.array([
                self.direction[0] * cos_a - self.direction[1] * sin_a,
                self.direction[0] * sin_a + self.direction[1] * cos_a
            ])
            norm = np.linalg.norm(self.direction)
            self.direction /= norm if norm > 0 else np.array([1.0, 0.0])
            self.position = (self.position + self.direction * self.cruise_speed) % self.space.size