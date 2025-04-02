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
        self.position = position
        self.speed = speed
        self.direction = direction
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.neighbors = []

    def step(self):
        """Get the Boid's neighbors, compute the new vector, and move accordingly."""
        neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
        self.neighbors = [n for n in neighbors if n is not self]

        # If no neighbors, maintain current direction
        if not neighbors:
            self.position += self.direction * self.speed
            return

        delta = self.space.calculate_difference_vector(self.position, agents=neighbors)

        cohere_vector = delta.sum(axis=0) * self.cohere_factor
        separation_vector = (
            -1 * delta[distances < self.separation].sum(axis=0) * self.separate_factor
        )
        match_vector = (
            np.asarray([n.direction for n in neighbors]).sum(axis=0) * self.match_factor
        )

        # Update direction based on the three behaviors
        self.direction += (cohere_vector + separation_vector + match_vector) / len(
            neighbors
        )

        # Normalize direction vector
        self.direction /= np.linalg.norm(self.direction)

        # Move boid
        self.position += self.direction * self.speed


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
        little_boid_neighbors = [
            (n, d) for n, d in zip(neighbors, distances) if isinstance(n, Prey)
        ]

        if little_boid_neighbors:
            # Find the closest Prey
            closest_neighbor, min_dist = min(little_boid_neighbors, key=lambda item: item[1])

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


        # Move the agent by directly updating its position
        self.position += self.direction * self.speed