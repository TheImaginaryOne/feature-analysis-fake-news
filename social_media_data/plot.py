import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import math, argparse

sns.set_theme()

parser = argparse.ArgumentParser(description='Test the classifier.')
parser.add_argument('dataset', type=str)
args = parser.parse_args()

splits_dir = f'{args.dataset}/splits'
def get_labels():
    df1 = pd.read_csv(splits_dir + '/train_0.txt', header=None)
    df2 = pd.read_csv(splits_dir + '/test_0.txt', header=None)
    df3 = pd.read_csv(splits_dir + '/val.txt', header=None)

    df = pd.concat([df1, df2, df3], axis=0)
    df.columns = ['id', 'label']
    df['id'] = df.id.astype(str)
    df.set_index('id', inplace=True)
    return df

def assert_eq(a, b):
    assert a == b, f"{a} is not equal {b}"

def run():
    df = pd.read_csv(f"output/{args.dataset}_feats_all.csv")
    df['id'] = df.id.astype(str)
    df.set_index('id', inplace=True)

    labels_df = get_labels()

    #assert_eq(len(labels_df), len(df))

    df = df.join(labels_df, on='id', how='inner') # discards some of the rows w/out a label
    #df['id'] = df.index
    print(df)
    a = ['image_neg', 'image_neu', 'image_pos']
    b = ['text_neg', 'text_neu', 'text_pos', 'text_comp']
    c = ['retweet_count', 'favorite_count']
    d = ['user_followers_count', 'user_friends_count', 'user_listed_count', 'user_statuses_count', 'user_age_days', 'user_favourites_count', 'user_verified']

    cols = a+b+c+d
    grid_colors = [0] * len(a) + [1] * len(b) + [2] * len(c) + [3] * len(d)
# 
    #print(proc_df)

    #proc_df['value'] = proc_df['value'].astype(np.float64)

    width = 4

    palette = sns.color_palette('pastel')

    fig, axes = plt.subplots(math.ceil(len(cols) / width), width, figsize=(12, 10))
    df['label'] = df['label'].map(lambda x: ['fake', 'real'][x])
    for x in range(width):
        for y in range(math.ceil(len(cols) / width)):
            i = x + y * width
            if i >= len(cols):
                axes[(y, x)].remove()
            else:
                col = cols[i]
                if col == "user_verified":
                    sns.countplot(data=df, x=df['label'].astype(str), hue=col, ax=axes[(y, x)])
                    #sns.move_legend(axes[(y,x)], bbox_to_anchor=(1.00, 0.00), loc='lower left') # move legend
                else:
                    sns.boxplot(data=df, x=df['label'].astype(str), y=col, ax=axes[(y, x)])
                axes[(y, x)].set_facecolor(palette[grid_colors[i]])
    fig.tight_layout()
    # avoid plot overlap
    patches = [Patch(color=palette[i], label=cat) for i, cat in enumerate(['image', 'text', 'behav', 'meta'])]
    fig.legend(handles=patches, bbox_to_anchor=(1.0, 0.5))
    plt.savefig(f'output/{args.dataset}_comparisons.png', box_inches='tight')

    fig2, axes = plt.subplots(1, 2, figsize=(15, 15))

    sns.histplot(data=df[df['label']==1], x='favorite_count', y='user_age_days', ax=axes[0], bins=100)
    sns.histplot(data=df[df['label']==0], x='favorite_count', y='user_age_days', ax=axes[1], bins=100)
    plt.savefig('output/pairplotmediaeval.png')

    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
    arr = pd.DataFrame(pipeline.fit_transform(df[cols]))
    #print(df['label'])
    arr['label'] = df.reset_index()['label']
    #print(arr)

    sns.scatterplot(data=arr, x=0, y=1, hue='label', ax=axes, sizes=(20, 20), size='label', alpha=0.2)
    plt.savefig('output/pairplotmediaeval.png')

    #print(df[['label']].value_counts(normalize=True))
    #X, y = df[cols], df['label']
    #rf = RandomForestClassifier()
    #print(cross_val_score(rf, X, y, cv=5, scoring='accuracy'))
    #======
run()
