import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('results.csv')

stats = data.groupby(['param_set', 'prey_population_size'])['pred_steps'].agg(['mean', 'std', 'count']).reset_index()
stats['se'] = stats['std'] / np.sqrt(stats['count'])

pivot = stats.pivot(index='prey_population_size', columns='param_set', values='mean')
errors = stats.pivot(index='prey_population_size', columns='param_set', values='se')

ax = pivot.plot(kind='bar', yerr=errors, capsize=5, rot=0)
ax.set_xlabel('Prey Population (2 trials each)')
ax.set_ylabel('Steps for all Predators to eat 10 Prey')
ax.set_title('Predator Performance vs Prey Behavior for Different Prey Populations')
plt.legend(title='Prey Behavior')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('plot.png')