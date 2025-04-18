# model.py
import os
import sys
sys.path.insert(0, os.path.abspath("../../../..")) # Adjust path if needed
from utils import get

import numpy as np
from mesa import Model
# Import both agent types
from agents import Prey, Predator # Import from local agents.py
from mesa.experimental.continuous_space import ContinuousSpace

class BoidFlockers(Model):
    """Flocker model class. Handles agent creation, placement and scheduling."""

    def __init__(
        self,
        Prey_population_size=get('prey_population_size'),
        Predator_population_size=get('predator_population_size'),
        width=get('width'),
        height=get('height'),
        speed=get('speed'),         # Speed for Preys
        vision=get('vision'),       # Vision for Preys
        separation=get('separation'),    # Separation for Preys
        cohere=get('cohere'),
        separate=get('separate'),
        match=get('match'),
        predator_speed=get('predator_speed'), # Specific speed for Predators
        predator_vision=get('predator_vision'), # Specific vision for Predators
        seed=None,
        threshold = get('threshold'),
        predator_introduce_time = get('predator_introduce_time')
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

        self.prey_to_remove = []
        self.threshold=threshold
        self.step_count = 0
        self.running = True
        self.predators = []
        self.predator_introduce_time = predator_introduce_time

        # Set up the space
        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=True,
            random=self.random,
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

        self.finish_time = None

    def step(self):
        self.step_count += 1

        if self.step_count == self.predator_introduce_time:
            for i in range(self.Predator_population_size):
                position = self.rng.random(size=2) * self.space.size
                direction = self.rng.uniform(-1, 1, size=2)
                pred = Predator(
                    model=self,
                    space=self.space,
                    position=position,
                    direction=direction,
                    speed=self.predator_speed,
                    vision=self.predator_vision,
                    threshold=self.threshold
                )
                self.predators.append(pred)

        self.agents.shuffle_do("step")

        if self.predators and all(p.eaten_count >= self.threshold for p in self.predators):
            self.finish_time = self.step_count
            self.running = False

        for prey in self.prey_to_remove:
            self.space._remove_agent(prey)
            self.agents.remove(prey)
        self.prey_to_remove.clear()
