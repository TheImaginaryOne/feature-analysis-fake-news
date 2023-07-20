import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("multimodal_train.tsv", sep='\t')

ax = sns.histplot(hue="2_way_label", x="score", bins=50, data=df, element="step", fill=False, log_scale=(False, True))

plt.savefig("scores.png")
