"""A Boid (bird-oid) agent for implementing Craig Reynolds's Boids flocking model.

This implementation uses numpy arrays to represent vectors for efficient computation
of flocking behavior.
"""

import numpy as np

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
        speed=1,
        direction=(1, 1),
        vision=1,
        separation=1,
        cohere=0.03,
        separate=0.015,
        match=0.05,
        escape=0.1
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
            escape: Relative importance of fleeing nearby predators (default: 0.1)
        """
        super().__init__(space, model)
        self.position = np.array(position) # Ensure position is numpy array
        self.speed = speed
        # Normalize initial direction
        norm = np.linalg.norm(direction)
        self.direction = direction / norm if norm > 0 else np.array([1.0, 0.0])
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.escape_factor = escape # Store the escape factor
        self.neighbors = []
        self.predators_nearby = [] # Optional: store nearby predators for visualization/debugging

    def step(self):
        """Get neighbors, calculate flocking and escape vectors, and move."""
        all_neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
        # Store all neighbors (excluding self) for potential future use
        self.neighbors = [n for n in all_neighbors if n is not self]

        # Separate neighbors into Prey (for flocking) and Predators (for escaping)
        flock_neighbors = [n for n in self.neighbors if isinstance(n, Prey)]
        predator_neighbors = [n for n in self.neighbors if isinstance(n, Predator)]
        self.predators_nearby = predator_neighbors # Store for optional use

        # --- Calculate Flocking Vectors (based on Prey neighbors) ---
        cohere_vector = np.zeros(2)
        separate_vector = np.zeros(2)
        match_vector = np.zeros(2)

        if flock_neighbors:
            flock_neighbor_positions = np.array([n.position for n in flock_neighbors])
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
            flocking_influence_count = len(flock_neighbors)

        # --- Calculate Escape Vector (based on Predator neighbors) ---
        escape_vector = np.zeros(2)
        if predator_neighbors:
            # Find the single closest predator
            predator_positions = np.array([p.position for p in predator_neighbors])
            delta_predators = self.space.calculate_difference_vector(self.position, agents=predator_neighbors)
            predator_distances = np.linalg.norm(delta_predators, axis=1)

            closest_predator_idx = np.argmin(predator_distances)
            # Vector pointing FROM predator TO self (use delta which is self.pos - neighbor.pos)
            vector_away_from_closest = delta_predators[closest_predator_idx]

            # Normalize and scale the escape vector
            norm = np.linalg.norm(vector_away_from_closest)
            if norm > 0:
                escape_vector = (vector_away_from_closest / norm) * self.escape_factor

        # --- Combine Vectors and Update Direction ---
        flocking_vector = cohere_vector + separate_vector + match_vector

        # Calculate the target direction purely based on current influences
        # In 1-on-1, flocking_vector is zero, so target_direction_vector equals escape_vector
        target_direction_vector = flocking_vector + escape_vector

        # Normalize the target direction vector IF it has magnitude
        norm = np.linalg.norm(target_direction_vector)
        if norm > 0:
            # If there are influences, SET (=) direction towards the normalized target
            self.direction = target_direction_vector / norm
        # else: If target_direction_vector is zero, agent keeps its previous direction.

        # --- Move Agent ---
        # (Movement code using self.position = ... % self.space.size remains the same)
        new_position = self.position + self.direction * self.speed
        self.position = new_position % self.space.size


class Predator(ContinuousSpaceAgent):
    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        speed=0.5, # Maybe make them slower?
        direction=(1, 0),
        vision=15 # Maybe give them different vision?
    ):
        super().__init__(space, model)
        self.position = np.array(position) # Ensure position is numpy array
        self.speed = speed
        # Normalize initial direction
        norm = np.linalg.norm(direction)
        self.direction = direction / norm if norm > 0 else np.array([1.0, 0.0])
        self.vision = vision

    def step(self):
        """Find the nearest Prey and move towards it."""
        # Find all neighbors within vision radius
        neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)

        # Filter for Prey agents
        prey_neighbors = [
            (n, d) for n, d in zip(neighbors, distances) if isinstance(n, Prey)
        ]

        if prey_neighbors:
            # Find the closest Prey
            closest_neighbor, min_dist = min(prey_neighbors, key=lambda item: item[1])

            # Calculate direction vector towards the closest neighbor
            # Use space.calculate_difference_vector for torus geometry if needed,
            # but for target direction, direct vector is often sufficient.
            # Let's use the direct vector for simplicity here. If torus effects are weird, use calculate_difference_vector.
            target_vector = closest_neighbor.position - self.position
            # Handle torus wrap-around for direction calculation if necessary using space.calculate_difference_vector
            # target_vector = self.space.calculate_difference_vector(closest_neighbor.position, self.position)[0] # Example if using space method

            # Normalize the target vector to get direction
            norm = np.linalg.norm(target_vector)
            if norm > 0:
                self.direction = target_vector / norm
            # else: keep current direction if already at the target? Or random? Keep current.

        else:
            # No Preys nearby: Optionally add random wander, or just continue
            # Let's add a small random perturbation to direction to avoid getting stuck
            random_turn_angle = self.model.random.uniform(-0.1, 0.1) # Small random turn
            cos_a, sin_a = np.cos(random_turn_angle), np.sin(random_turn_angle)
            self.direction = np.array([
                self.direction[0] * cos_a - self.direction[1] * sin_a,
                self.direction[0] * sin_a + self.direction[1] * cos_a
            ])
            # Re-normalize just in case
            norm = np.linalg.norm(self.direction)
            self.direction /= norm if norm > 0 else np.array([1.0, 0.0])

        # Calculate new position
        new_position = self.position + self.direction * self.speed
        # Manually wrap coordinates using modulo and space size
        self.position = new_position % self.space.size