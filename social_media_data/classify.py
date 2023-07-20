import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
#from hypopt import GridSearch # grid search once
import argparse

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

from itertools import chain, combinations                                                                                                                                                     
from sklearn.dummy import DummyClassifier
from utils import merge_all_labels, get_feats, merge_dfs

import functools

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

parser = argparse.ArgumentParser(description='Test the classifier.')
parser.add_argument('dataset', type=str)
parser.add_argument('--useownsplits', action='store_true')
parser.add_argument('--oversample', action='store_true')
parser.add_argument('--balance', action='store_true')
args = parser.parse_args()
dataset = args.dataset

output_dir = 'output'
splits_dir = f'{args.dataset}/splits'

a = ['image_neg', 'image_neu', 'image_pos']
b = ['text_neg', 'text_neu', 'text_pos', 'text_comp']
c = ['retweet_count', 'favorite_count']
d = ['user_verified', 'user_followers_count', 'user_friends_count', 'user_listed_count', 'user_statuses_count', 'user_age_days', 'user_favourites_count']

feats = a+b+c+d
target = 'label'
subset = feats + [target]

seed = 141

def grab_features(df, chosen_feats):
    X = df[chosen_feats]
    y = df[target]
    return X, y

def run(df_tr, df_te, chosen_feats, name, split, is_dummy=False):
    print("=== features: ", chosen_feats)

    X_tr, y_tr = grab_features(df_tr, chosen_feats)
    X_te, y_te = grab_features(df_te, chosen_feats)
    #X_te, y_te = grab_features(df_te, chosen_feats)
    #print(len(X_tr), len(X_val), len(X_te))
    #print(len(y_tr), len(y_val), len(y_te))

    # Workaround to use presplitted validation/train sets
    #X_concat = np.concatenate((X_val, X_tr), axis=0)
    #y_concat = np.concatenate((y_val, y_tr), axis=0)

    def train(name, grid, model):
        print(f"-- model: {name} ---")
        if args.oversample:
            clf = make_pipeline(
                    RandomOverSampler(random_state=seed),
                    GridSearchCV(estimator=model, param_grid=grid, cv=split, verbose=1, n_jobs=1)
                    )
        else:
            clf = make_pipeline(
                    GridSearchCV(estimator=model, param_grid=grid, cv=split, verbose=1, n_jobs=1)
                    )

        #print("Traning...")
        #clf.fit(X_concat, y_concat)
        clf.fit(X_tr, y_tr)
        #print(f"GridSearch results: {clf.get_param_scores()}")
        #print("Testing...")
        y_pr = clf.predict(X_te)
        #print(y_pr)
        accuracy = accuracy_score(y_te, y_pr)
        recall = recall_score(y_te, y_pr)
        precision = precision_score(y_te, y_pr)
        cm = confusion_matrix(y_te, y_pr, labels=[0,1])
        f1 = f1_score(y_te, y_pr)
        #print("accuracy:", accuracy)
        return {'tn': cm[0, 0], 'fp': cm[0, 1],
                        'fn': cm[1, 0], 'tp': cm[1, 1]}

    # rf
    grid = {'n_jobs': [16], 'min_samples_split': [2, 10, 100], 'min_samples_leaf': [20, 100], 'random_state': [seed], 'n_estimators': [100]}

    if is_dummy:
        data = train("dummy", {}, DummyClassifier())
    else:
        data = train("rf", grid, RandomForestClassifier())
    data['fold'] = name

    return data


results_list = []
# the dataset already has five splits
def run_all(chosen_feats, useownsplits, is_dummy=False):
    print("Use own splits?", useownsplits)
    feats_df = get_feats(output_dir, dataset)
    results_list = []
    if useownsplits:
        df_all = merge_all_labels(feats_df, args.balance, seed, output_dir, dataset)
        from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
        kfolds = RepeatedStratifiedKFold(n_splits=5, random_state=seed, n_repeats=8)
        kfold_splits = kfolds.split(df_all, df_all['label'])

    n_iters = 5 * 4 if useownsplits else 5
    for split_n in range(n_iters):
        if useownsplits:
            train_i, test_i = next(kfold_splits)
            df_te = df_all.iloc[test_i]
            df_tr_fold = df_all.iloc[train_i]
            #inner_split = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)
            inner_split = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)
        else:
            labels_tr = pd.read_csv(f"{splits_dir}/train_{split_n}.txt", header=None)
            labels_te = pd.read_csv(f"{splits_dir}/test_{split_n}.txt", header=None)
            labels_val = pd.read_csv(f"{splits_dir}/val.txt", header=None)
            print("- Train count:", len(labels_tr), "Test:", len(labels_te), "val:", len(labels_val))

            for df in [labels_tr, labels_te, labels_val]:
                df.columns = ['id', 'label']
                df['id'] = df['id'].astype(str)
                df.set_index('id', inplace=True)
    
            df_tr = merge_dfs(labels_tr, feats_df)
            df_te = merge_dfs(labels_te, feats_df)
            df_val = merge_dfs(labels_val, feats_df)
            assert len(df_tr) == len(labels_tr)
            assert len(df_te) == len(labels_te)
            assert len(df_val) == len(labels_val)
            # fixed split
            inner_split = PredefinedSplit(test_fold=[1 for _ in range(len(df_val))] + [-1 for _ in range(len(df_tr))])
            df_tr_fold = pd.concat([df_val, df_tr])
        #print("-- train_df: -- \n", df_tr)
        #print("-- test_df: -- \n", df_te)
        #print(len(df_tr), len(df_te), len(df_va
        #print(df_tr)
        #print(df_te['truth_label'].value_counts(normalize=True))
    
        results = run(df_tr_fold, df_te, chosen_feats, f"{split_n}", inner_split, is_dummy)
        results_list.append(results)

    df = pd.DataFrame(results_list)

    return df

results = []

# run dummy
#result = run_all([], args.balanced, True) # join the lists to form one array
#result['name'] = 'dummy'
#results.append(result)

for chosen_feats, names in zip(powerset([a,b,c,d]), powerset(['image', 'text', 'behav', 'meta']) ):
    name = "+".join(names) # concat
    print(name)
    result = run_all(sum(chosen_feats, []), args.useownsplits) # join the lists to form one array
    result['name'] = "+".join(names)
    results.append(result)


pd.concat(results, ignore_index=True, axis=0).to_csv(f"output/{dataset}_{'oversample_' if args.oversample else ''}{'balance_' if args.balance else ''}clf_results.csv")
