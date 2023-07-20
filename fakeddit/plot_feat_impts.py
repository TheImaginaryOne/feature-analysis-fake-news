import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#feat_impts = [0.0242075, 0.02541708, 0.03104469, 0.01316271, 0.02675186, 0.01616032, 0.0181214,  0.33433489, 0.0983057,  0.41249385]

sns.set_theme()

fi_df = pd.read_csv('output/clf_feat_impts.csv')

last_row = fi_df.iloc[len(fi_df) - 1]
 
feat_impts = [float(s) for s in last_row.feat_impt[1:-1].split()]
feats = [s[1:-1] for s in last_row.feats[1:-1].split()]

def plot_bar(labels, values):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    bar = sns.barplot(x=pd.Series(labels), y=pd.Series(values), ax=ax)
    bar.bar_label(bar.containers[0])

    plt.savefig('output/feat_impts.png', bbox_inches="tight")

plot_bar(feat_impts, feats)
