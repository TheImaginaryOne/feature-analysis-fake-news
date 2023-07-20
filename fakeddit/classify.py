import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.target_encoder import TargetEncoder

#from hypopt import GridSearch # grid search once

from itertools import chain, combinations
import functools

def powerset(iterable):
    """ exclude empty set """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))



a = ['image_neg','image_neu','image_pos']
b = ['title_neg','title_neu','title_pos','title_comp']#,'avg_comp_comment','avg_neg_comment','avg_neu_comment','avg_pos_comment','max_comp_comment','max_neg_comment','max_neu_comment','max_pos_comment','min_comp_comment','min_neg_comment','min_neu_comment','min_pos_comment']

c = ['upvote_ratio','score','num_comments']
d = ['domain', 'author']
feats = a + b + c + d #+ 

target = '2_way_label'
subset = feats + [target]

def filter_df(df):
    #df = df[subset].dropna()
    df = df[~df['subreddit'].isin(['psbattle_artwork', 'photoshopbattles'])]#, 'subredditsimulator'])]
    return df

def grab_features(df, chosen_feats):
    X = df[chosen_feats]
    y = df[target]
    return X, y

def run(df_tr, df_val, df_te, chosen_feats, name):
    print("=== features: ", chosen_feats)

    # test a fake news prediction with these features.
    # avg_neg_comment,avg_neu_comment,avg_pos_comment
    #X = df[['upvote_ratio']]
    #X = df[['num_comments','score','upvote_ratio']]
    X_tr, y_tr = grab_features(df_tr, chosen_feats)
    X_val, y_val = grab_features(df_val, chosen_feats)
    X_te, y_te = grab_features(df_te, chosen_feats)
    seed = 145
    #seed = 83831

    # Workaround to use presplitted validation/train sets
    X_concat = pd.concat([X_val, X_tr], axis=0)
    assert len(X_val) + len(X_tr) == len(X_concat)
    y_concat = pd.concat([y_val, y_tr], axis=0)
    assert len(y_val) + len(y_tr) == len(y_concat)

    split = PredefinedSplit(test_fold=[1 for _ in range(len(df_val))] + [-1 for _ in range(len(df_tr))])

    # Column transformer re orders feature names so we must ensure the order is as expected
    reordered_feats = d + [f for f in chosen_feats if (f not in d)]

    should_transform = set(d).issubset(set(chosen_feats))

    def train(name, grid, model):
        print(f"-- model: {name} ---")

        transformers = []
        if should_transform:
            # ordinal
            transformers = [('encode', TargetEncoder(), d)]

        col_trans = ColumnTransformer(transformers, remainder='passthrough')
            
        clf = GridSearchCV(param_grid=grid, cv=split, verbose=1, n_jobs=1, estimator=Pipeline([
            ('col_trans', col_trans),
            ('model', model)
            ]))


        #print("Traning...")
        clf.fit(X_concat, y_concat)
        #print(f"GridSearch results: {clf.get_param_scores()}")
        #print("Testing...")
        y_pr = clf.predict(X_te)
        print(y_pr)
        accuracy = accuracy_score(y_te, y_pr)
        recall = recall_score(y_te, y_pr, pos_label=0) # pos_label is relative to the "fake" class
        precision = precision_score(y_te, y_pr, pos_label=0)
        matrix = confusion_matrix(y_te, y_pr)
        f1 = f1_score(y_te, y_pr, pos_label=0)
        print("accuracy:", accuracy)
        return (clf, clf.best_estimator_['model'], accuracy, recall, precision, matrix, f1)

    # logistic regression
    #grid = {'C': [0.001, 0.01, 0.1, 1.0, 5.0, 20.0, 200.0], 'penalty': ['l2'], 'random_state': [seed]}
    #best_model, score = train("logreg", grid, LogisticRegression())
    #print("feature importances: ", best_model.coef_)

    # rf
    grid = {'model__n_jobs': [16], 'model__min_samples_split': [2, 20, 100], 'model__min_samples_leaf': [2, 20, 100], 'model__random_state': [seed], 'model__n_estimators': [50]}
    if should_transform:
        grid['col_trans__encode__smoothing'] = [1., 10.0, 100.0]
        grid['col_trans__encode__min_samples_leaf'] = [1., 10.0, 100.0]

    #grid = {'learning_rate':[0.1, 0.4], 'max_iter':[100, 300]}
    clf, best_model, rf_score, recall, precision, matrix, f1 = train("randforest", grid, RandomForestClassifier())
    feat_impts = np.array(best_model.feature_importances_) # TODO
    #print("feature importances:", best_model.feature_importances_)

    return {'name': name, 'accuracy': rf_score, 'precision': precision, 'recall': recall, 'f1': f1}, \
        {'feat_impt': np.array2string(feat_impts), 'name': name, 'feats': np.array2string(np.array(reordered_feats)) }
    #, 'confusion_matrix': matrix}
    #
    #grid = {'C': [0.2, 0.5, 1.0, 5.0]}
    #best_model = train("svm", grid, SVC())

# Run the experiment
train_df = pd.read_csv("output/merged.csv")
test_df = pd.read_csv("output/merged_test.csv")
val_df = pd.read_csv("output/merged_validate.csv")
train_df = filter_df(train_df)
test_df = filter_df(test_df)
val_df = filter_df(val_df)
print(f"After filtering, training data got: {len(train_df)}")
print(f"After filtering, testing data got: {len(test_df)}")
print(f"After filtering, val data got: {len(val_df)}")
#print(train_df.isna().sum())

print(test_df['2_way_label'].value_counts(normalize = True))

#print((train_df == np.inf).sum())
results = []

feat_impts = []

for chosen_feats, names in zip(powerset([a, b, c, d]), \
        powerset(["image", "text", "behav", "meta"])):
    chosen_feats = functools.reduce(lambda x, y: x+y, chosen_feats)
    name = functools.reduce(lambda x, y: x+"+"+y, names)
    result, feat_impt_result = run(train_df, val_df, test_df, chosen_feats, name)
    results.append(result)
    feat_impts.append(feat_impt_result)

pd.DataFrame(results).to_csv("output/clf_results.csv", index=False)
pd.DataFrame(feat_impts).to_csv("output/clf_feat_impts.csv", index=False)
