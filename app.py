import os
import sys
sys.path.insert(0, os.path.abspath("../../../.."))  # Adjust path if needed
from utils import get
import matplotlib.pyplot as plt
from mesa.visualization import Slider, SolaraViz, make_space_component
from mesa.visualization.utils import update_counter
import solara
from model import BoidFlockers  # Import from local model.py
from agents import Prey, Predator  # Import from local agents.py

def bar_draw(model):
    fig, ax = plt.subplots(figsize=(4, 3))
    if model.predators:  # Check if there are any predators
        counts = [p.eaten_count for p in model.predators]
        labels = [f"P{i}" for i in range(len(counts))]
        ax.bar(labels, counts)
        ax.set_ylim(0, model.threshold)  # Keep y-axis consistent
    else:
        ax.text(0.5, 0.5, "No predators", horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_xlabel("Predator")
    ax.set_ylabel("Prey eaten")
    ax.set_title("Eaten per Predator")
    plt.tight_layout()
    return fig

@solara.component
def BarGraph(model):
    update_counter.get()  # Ensure component updates with model changes
    fig = bar_draw(model)
    solara.FigureMatplotlib(fig)


# Updated drawing function
def boid_draw(agent):
    if isinstance(agent, Prey):
        # Render Prey (e.g., green circle)
        return {"color": "blue", "size": 15, "marker": "o"}
    elif isinstance(agent, Predator):
        # Render Predator (e.g., large red square)
        return {"color": "red", "size": 45, "marker": "o"}


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "Prey_population_size": Slider(
        label="Number of Prey",
        value=get('prey_population_size'),
        min=1,
        max=300,
        step=10,
    ),
    "Predator_population_size": Slider(
        label="Number of Predators",
        value=get('predator_population_size'), # Default value from model
        min=0,
        max=5,
        step=1,
    ),
    "width": get('width'), # Keep width and height as they are likely used by space
    "height": get('height'),
    # Parameter for Prey speed, updated label
    "speed": Slider(
        label="Speed of Prey",
        value=get('speed'), # Default value from user's previous app.py (matches model default if speed=1, adjust if needed)
        min=1,
        max=10,
        step=0.5,
    ),
    # Parameter for Prey vision, updated label
    "vision": Slider(
        label="Vision of Prey (radius)",
        value=get('vision'), # Default value from model
        min=1,
        max=30,
        step=1,
    ),
    # Parameter for Prey separation, updated label
    "separation": Slider(
        label="Minimum Separation (Prey)",
        value=get('separation'), # Default value from model
        min=1,
        max=10,
        step=1,
    ),
     # Added parameter for predator speed
    "predator_speed": Slider(
        label="Speed of Predators",
        value=get('predator_speed'), # Default value from user's previous app.py (matches model default if predator_speed=0.5, adjust if needed)
        min=0.1,
        max=10,
        step=0.1,
    ),
    # Added parameter for predator vision
    "predator_vision": Slider(
        label="Vision of Predator (radius)",
        value=get('predator_vision'), # Default value from model
        min=1,
        max=50,
        step=1,
    ),
    "threshold": Slider(
        label="Prey eaten per Predator",
        value=get('threshold'),
        min=1,
        max=50,
        step=1,
    ),
    # Flocking parameters (for Prey), ensure keys match model __init__
    "cohere": Slider(label="Cohesion Factor (Prey)", value=get('cohere'), min=0.0, max=1.0, step=0.05),
    "separate": Slider(label="Separation Factor (Prey)", value=get('separate'), min=0.0, max=1.0, step=0.05),
    "match": Slider(label="Alignment Factor (Prey)", value=get('match'), min=0.0, max=1.0, step=0.05),
}

model = BoidFlockers()

page = SolaraViz(
    model,
    components=[
        make_space_component(agent_portrayal=boid_draw, backend="matplotlib", figsize=(20,20)),
        BarGraph     
    ],
    model_params=model_params,
    name="Predator-Prey Flocking Model",
)

page  # noqa