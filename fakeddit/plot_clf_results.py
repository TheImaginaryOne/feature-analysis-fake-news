import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

sns.set_theme()

results_df = pd.read_csv('output/clf_results.csv').sort_values(by='accuracy', ascending=False)

# print tables
print(results_df.to_latex(longtable=False, float_format="%.4f"))

results_df = pd.melt(results_df, id_vars='name', var_name='measure')

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.barplot(data=results_df, y='name', x='value', hue='measure', ax=ax)

plt.savefig('output/clf_results_plot.png', bbox_inches="tight")
