import pandas as pd
from pathlib import Path
from model import BoidFlockers
from utils import get 

N_TRIALS   = 5       # how many random seeds per parameter set
MAX_STEPS  = 5000    # failsafe upper bound so runs don’t go forever

PREY_POPULATIONS = [100, 250, 500, 1000]

PARAM_SETS = [
    dict(name="flock", separation=1, cohere=0.05, match=1, prey_population_size=pop)
    for pop in PREY_POPULATIONS
] + [
    dict(name="no_flock", separation=0, cohere=0, match=0, prey_population_size=pop)
    for pop in PREY_POPULATIONS
]

def run_once(seed: int, sweep: dict) -> dict:
    """Instantiate a model, advance until done / MAX_STEPS, return a dict of metrics."""
    m = BoidFlockers(
        seed=seed,
        separation=sweep["separation"],
        cohere=sweep["cohere"],
        match=sweep["match"],
        Prey_population_size=sweep["prey_population_size"],
    )

    for _ in range(MAX_STEPS):
        m.step()
        if not m.running:
            break

    predator_steps = m.finish_time - get('predator_introduce_time') if m.finish_time else MAX_STEPS

    return {
        "param_set": sweep["name"],
        "prey_population_size": sweep["prey_population_size"],
        "finished": m.finish_time is not None,
        "pred_steps": predator_steps,
    }

def main() -> None:
    records = []
    for sweep in PARAM_SETS:
        for trial in range(N_TRIALS):
            print(sweep, trial)
            records.append(run_once(trial, sweep))

    df = pd.DataFrame(records)
    df.to_csv("results.csv", index=False)

    print("\nPrediction steps summary (lower is faster):")
    print(df.groupby(["param_set", "prey_population_size"])["pred_steps"].describe())

    print(f"\nRaw data saved to {Path('results.csv').resolve()}")

if __name__ == "__main__":
    main()