# app.py
import os
import sys
sys.path.insert(0, os.path.abspath("../../../..")) # Adjust path if needed

# Make sure it imports the updated model definition
from model import BoidFlockers # Import from local model.py
from mesa.visualization import Slider, SolaraViz, make_space_component
import numpy as np # Import numpy if needed for any potential logic later

from agents import Prey, Predator # Import from local agents.py


# Updated drawing function
def boid_draw(agent):
    if isinstance(agent, Prey):
        # Render Prey (e.g., green circle)
        return {"color": "green", "size": 20, "marker": "o"}
    elif isinstance(agent, Predator):
        # Render Predator (e.g., large red square)
        return {"color": "red", "size": 40, "marker": "s"}
    else:
        # Fallback for any other agent type
        return {"color": "gray", "size": 10}

# Update model parameters for Solara UI to match model.__init__
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    # Renamed parameter and updated label
    "Prey_population_size": Slider(
        label="Number of Prey",
        value=100,
        min=10,
        max=200,
        step=10,
    ),
    # Added parameter for predators
    "Predator_population_size": Slider(
        label="Number of Predators",
        value=10, # Default value from model
        min=0,
        max=50,
        step=1,
    ),
    "width": 100, # Keep width and height as they are likely used by space
    "height": 100,
    # Parameter for Prey speed, updated label
    "speed": Slider(
        label="Speed of Prey",
        value=5, # Default value from user's previous app.py (matches model default if speed=1, adjust if needed)
        min=1,
        max=20,
        step=1,
    ),
    # Parameter for Prey vision, updated label
    "vision": Slider(
        label="Vision of Prey (radius)",
        value=10, # Default value from model
        min=1,
        max=50,
        step=1,
    ),
    # Parameter for Prey separation, updated label
    "separation": Slider(
        label="Minimum Separation (Prey)",
        value=2, # Default value from model
        min=1,
        max=20,
        step=1,
    ),
     # Added parameter for predator speed
    "predator_speed": Slider(
        label="Speed of Predators",
        value=2, # Default value from user's previous app.py (matches model default if predator_speed=0.5, adjust if needed)
        min=0.1,
        max=10,
        step=0.1,
    ),
    # Added parameter for predator vision
    "predator_vision": Slider(
        label="Vision of Predator (radius)",
        value=15, # Default value from model
        min=1,
        max=50,
        step=1,
    ),
    # Flocking parameters (for Prey), ensure keys match model __init__
    "cohere": Slider(label="Cohesion Factor (Prey)", value=0.03, min=0.0, max=0.1, step=0.005),
    "separate": Slider(label="Separation Factor (Prey)", value=0.015, min=0.0, max=0.1, step=0.005),
    "match": Slider(label="Alignment Factor (Prey)", value=0.05, min=0.0, max=0.2, step=0.01),
}

# Instantiate the model (uses defaults from model.py unless overridden by UI)
model = BoidFlockers()

page = SolaraViz(
    model,
    components=[make_space_component(agent_portrayal=boid_draw, backend="matplotlib")],
    model_params=model_params,
    name="Predator-Prey Flocking Model", # Updated name
)

page  # noqa
