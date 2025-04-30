import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NUM_PREDS = 5

data = pd.read_csv(f'results_{NUM_PREDS}_predators.csv')

# compute means, SEs
stats = (
    data
    .groupby(['param_set', 'prey_population_size'])['pred_steps']
    .agg(['mean','std','count'])
    .reset_index()
)
stats['se'] = stats['std'] / np.sqrt(stats['count'])

# pivot for bar‑plot
pivot = stats.pivot(index='prey_population_size',
                    columns='param_set',
                    values='mean')
errors = stats.pivot(index='prey_population_size',
                     columns='param_set',
                     values='se')

# map each series to a color
color_map = {
    # non‑schooling = blues
    "No Schooling + Direct Flee":        "#0D47A1",  # dark blue
    "No Schooling + Random Scatter Flee":"#64B5F6",  # light blue
    # schooling = oranges
    "Schooling + Direct Flee":           "#D84315",  # dark orange
    "Schooling + Random Scatter Flee":   "#FF8A65",  # light orange
}
colors = [ color_map[col] for col in pivot.columns ]

fig, ax = plt.subplots(figsize=(12, 6))
pivot.plot(
    kind='bar',
    yerr=errors,
    capsize=5,
    rot=0,
    color=colors,
    ax=ax,
    width=0.8
)

ax.set_xlabel('Prey Population (n=5 trials)')
ax.set_ylabel('Steps until all predators eat 10 prey')
ax.set_title(f'{NUM_PREDS}-Predator Performance vs. Prey Behavior & Population Size')

ax.legend(
    title='Prey Behavior',
    loc='upper left',
    bbox_to_anchor=(1.02, 1)
)

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 0.8, 1])

plt.savefig(f'plot_{NUM_PREDS}_preds.png', bbox_inches='tight')