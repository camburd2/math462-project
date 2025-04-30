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

        prey_cruise_speed=get('prey_cruise_speed'),
        prey_burst_speed=get('prey_burst_speed'),
        prey_vision=get('prey_vision'),
        separation=get('separation'),
        cohere=get('cohere'),
        separate=get('separate'),
        match=get('match'),
        prey_burst_cooldown=get('prey_burst_cooldown'),
        prey_burst_length=get('prey_burst_length'),
        prey_rand_scatter=get('prey_rand_scatter'),

        predator_cruise_speed=get('predator_cruise_speed'),
        predator_burst_speed=get('predator_burst_speed'),
        predator_vision=get('predator_vision'),

        seed=None,
        threshold = get('threshold'),
        predator_introduce_time = get('predator_introduce_time'),
        eat_radius = get('eat_radius'),
        predator_burst_dist = get('predator_burst_dist'),
        predator_max_turn_angle = get('predator_max_turn_angle'),
        predator_burst_length = get('predator_burst_length')
    ):
        super().__init__(seed=seed)
        self.Prey_population_size = int(Prey_population_size)
        self.Predator_population_size = int(Predator_population_size)

        self.prey_cruise_speed = float(prey_cruise_speed)
        self.prey_burst_speed = float(prey_burst_speed)
        self.prey_vision = float(prey_vision)
        self.separation = float(separation)
        self.cohere = float(cohere)
        self.separate = float(separate)
        self.match = float(match)
        
        self.prey_burst_cooldown= int(prey_burst_cooldown)
        self.prey_burst_length= int(prey_burst_length)
        self.prey_rand_scatter= int(prey_rand_scatter)

        self.predator_cruise_speed = float(predator_cruise_speed)
        self.predator_burst_speed = float(predator_burst_speed)
        self.predator_burst_dist = float(predator_burst_dist)
        self.predator_vision = float(predator_vision)
        self.eat_radius = float(eat_radius)

        self.prey_to_remove = []
        self.threshold=threshold
        self.step_count = 0
        self.running = True
        self.predators = []
        self.predator_introduce_time = predator_introduce_time

        self.predator_max_turn_angle_rad = np.deg2rad(float(predator_max_turn_angle))
        self.predator_burst_length = float(predator_burst_length)


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
                cruise_speed=self.prey_cruise_speed,
                burst_speed=self.prey_burst_speed,
                vision=self.prey_vision,
                separation=self.separation,
                cohere=self.cohere,
                separate=self.separate,
                match=self.match,
                burst_cooldown=self.prey_burst_cooldown,
                burst_length=self.prey_burst_length,
                prey_rand_scatter=self.prey_rand_scatter               
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
                    cruise_speed=self.predator_cruise_speed,
                    burst_speed=self.predator_burst_speed,
                    vision=self.predator_vision,
                    threshold=self.threshold,
                    eat_radius=self.eat_radius,
                    burst_dist=self.predator_burst_dist,
                    max_turn_angle=self.predator_max_turn_angle_rad,
                    burst_length=self.predator_burst_length
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
