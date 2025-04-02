# model.py
import os
import sys
sys.path.insert(0, os.path.abspath("../../../..")) # Adjust path if needed

import numpy as np
from mesa import Model
# Import both agent types
from agents import Prey, Predator # Import from local agents.py
from mesa.experimental.continuous_space import ContinuousSpace

class BoidFlockers(Model):
    """Flocker model class. Handles agent creation, placement and scheduling."""

    # Add Predator_population_size parameter with a default
    def __init__(
        self,
        Prey_population_size=100, # Renamed for clarity
        Predator_population_size=10,     # Added new population
        width=100,
        height=100,
        speed=1,         # Speed for Preys
        vision=10,       # Vision for Preys
        separation=2,    # Separation for Preys
        cohere=0.03,
        separate=0.015,
        match=0.05,
        predator_speed=0.5, # Specific speed for Predators
        predator_vision=15, # Specific vision for Predators
        seed=None,
    ):
        super().__init__(seed=seed)
        self.Prey_population_size = Prey_population_size
        self.Predator_population_size = Predator_population_size
        # Store other parameters if needed by agents
        self.speed = speed
        self.vision = vision
        self.separation = separation
        self.cohere = cohere
        self.separate = separate
        self.match = match
        self.predator_speed = predator_speed
        self.predator_vision = predator_vision

        # Calculate total population for space allocation if needed by ContinuousSpace version
        total_population = Prey_population_size + Predator_population_size

        # Set up the space
        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=True,
            random=self.random,
            # n_agents=total_population, # Check if n_agents is strictly required/used by this version
        )

        # Create Prey agents
        for i in range(self.Prey_population_size):
            position = self.rng.random(size=2) * self.space.size
            direction = self.rng.uniform(-1, 1, size=2)
            agent = Prey(
                model=self,
                space=self.space,
                position=position,
                direction=direction,
                speed=self.speed,
                vision=self.vision,
                separation=self.separation,
                cohere=self.cohere,
                separate=self.separate,
                match=self.match,
            )
            #self.space.add_agent(agent)

        # Create Predator agents
        for i in range(self.Predator_population_size):
            position = self.rng.random(size=2) * self.space.size
            direction = self.rng.uniform(-1, 1, size=2)
            agent = Predator(
                model=self,
                space=self.space,
                position=position,
                direction=direction,
                speed=self.predator_speed,
                vision=self.predator_vision,
            )
            #self.space.add_agent(agent)

        # For tracking statistics (currently averages heading of ALL agents)
        self.average_heading = None
        # self.update_average_heading() # Calculate initial heading if needed

    def update_average_heading(self):
        """Calculate the average heading (direction) of all Prey agents."""
        # Filter for Preys only
        preys = [agent for agent in self.agents if isinstance(agent, Prey)]
        if not preys:
            self.average_heading = 0
            return

        headings = np.array([agent.direction for agent in preys])
        mean_heading = np.mean(headings, axis=0)
        # Protect against zero vector if all directions cancel out
        norm = np.linalg.norm(mean_heading)
        if norm > 0:
            # Calculate angle from the mean vector components
            self.average_heading = np.arctan2(mean_heading[1], mean_heading[0])
        else:
             self.average_heading = 0 # Or None, or NaN, depending on desired handling


    def step(self):
        """Run one step of the model."""
        # shuffle_do activates all agents regardless of type
        self.agents.shuffle_do("step")
        self.update_average_heading() # Update average heading of Preys
