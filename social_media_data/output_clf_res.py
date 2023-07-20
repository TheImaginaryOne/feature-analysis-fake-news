import pandas as pd
import argparse
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt

sns.set_theme()

parser = argparse.ArgumentParser(description='Test the classifier.')
parser.add_argument('dataset', type=str)
parser.add_argument('--oversample', action='store_true')
parser.add_argument('--balance', action='store_true')
args = parser.parse_args()
dataset = args.dataset

def calculate(df):
    ret_df = pd.DataFrame()
    ret_df['name'] = df['name']
    total = df.tn + df.fn + df.tp + df.fp
    ret_df['accuracy'] = (df.tn + df.tp) / total
    ret_df['precision'] = df.tn / (df.tn + df.fn)
    ret_df['recall'] = df.tn / (df.tn + df.fp)
    ret_df['f1'] = 2 / (1 / ret_df.precision + 1 / ret_df.recall)
    return ret_df

df_ = pd.read_csv(f"output/{dataset}_{'oversample_' if args.oversample else ''}{'balance_' if args.balance else ''}clf_results.csv", index_col=0)
df = calculate(df_)

grouped = df.groupby("name").agg(mean_acc=('accuracy', 'mean'))

# ===
df = df.join(grouped, on="name").sort_values("mean_acc")

cols = ['accuracy', 'precision', 'recall', 'f1']
kwargs = {**{f'mean_{x}': (x, 'mean') for x in cols}, **{f'stddev_{x}': (x, 'std') for x in cols}}

grouped_all = df.groupby("name").agg(**kwargs).sort_values("mean_accuracy")

print(tabulate(grouped_all[['mean_accuracy', 'mean_f1', 'mean_precision', 'mean_recall', 'stddev_accuracy']], headers="keys", floatfmt=".4f", tablefmt="latex"))
# ===

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.barplot(data=pd.melt(df, id_vars=['name', 'mean_acc']), x='value', y='name', hue='variable', ci=95, ax=ax, n_boot=10000)

plt.savefig(f"output/{dataset}_{'oversample_' if args.oversample else ''}{'balance_' if args.balance else ''}clf_results.png", bbox_inches='tight')
