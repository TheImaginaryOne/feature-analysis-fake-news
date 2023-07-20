import argparse
import pandas as pd

from imblearn.pipeline import Pipeline
from utils import merge_dfs, merge_all_labels, get_feats
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Test the classifier.')
parser.add_argument('dataset', type=str)
args = parser.parse_args()
dataset = args.dataset

output_dir = 'output'
seed = 141
target = 'label'

a = ['image_neg', 'image_neu', 'image_pos']
b = ['text_neg', 'text_neu', 'text_pos', 'text_comp']
c = ['retweet_count', 'favorite_count']
d = ['user_verified', 'user_followers_count', 'user_friends_count', 'user_listed_count', 'user_statuses_count', 'user_age_days', 'user_favourites_count']

feats = a+b+c+d

# get features by names
def grab_features(df: pd.DataFrame):
    X = df[feats]
    y = df[target]
    return X, y

def run():
    feats_df = get_feats(output_dir, dataset)
    results_list = []
    df_all = merge_all_labels(feats_df, True, seed, output_dir, dataset)

    return get_feat_impts(df_all)


def get_feat_impts(df_tr, is_dummy=False):
    #print("=== features: ", chosen_feats)

    X_tr, y_tr = grab_features(df_tr)

    model = RandomForestClassifier()

    # rf
    grid = {'n_jobs': [16], 'min_samples_split': [2, 10, 100], 'min_samples_leaf': [20, 100], 'random_state': [seed], 'n_estimators': [100]}

    split = ShuffleSplit(n_splits=1, train_size=0.8)

    clf = Pipeline(
            [('pre', RandomOverSampler(random_state=seed)),
            ('cv', GridSearchCV(estimator=model, param_grid=grid, cv=split, verbose=1, n_jobs=1))]
            )

    clf.fit(X_tr, y_tr)

    return clf['cv'].best_estimator_.feature_importances_

sns.set_theme()

fig, ax = plt.subplots()
fig.set_figwidth(8)
fig.set_figheight(5)
data = pd.DataFrame([feats, run()]).transpose()

chart = sns.barplot(data=data, x=1, y=0, ax=ax)
chart.bar_label(chart.containers[0])

plt.savefig(f"{output_dir}/{dataset}_feat_impts.png", bbox_inches='tight')