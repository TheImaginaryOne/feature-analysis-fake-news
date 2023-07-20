import pandas as pd

def merge_dfs(labels, feats):
    # the labels df has no header

    return feats.join(labels, on='id', how='inner')

def get_feats(output_dir, dataset):
    feats_df = pd.read_csv(f"{output_dir}/{dataset}_feats_all.csv")
    feats_df['id'] = feats_df['id'].astype(str)
    feats_df.set_index('id', inplace=True)
    return feats_df

from imblearn.under_sampling import RandomUnderSampler

# add in the list of the labels
def merge_all_labels(feats_df, balance, seed, output_dir, dataset):
    all_labels_df = pd.read_csv(f"{output_dir}/{dataset}_labels.csv")
    # balance the truth/false classes
    samp = RandomUnderSampler(random_state=seed)
    labels_df_bal = pd.DataFrame()
    # execute resampler
    if balance:
        ids, labels = samp.fit_resample(all_labels_df[['id']], all_labels_df['label'])
    else:
        ids, labels = all_labels_df[['id']], all_labels_df['label']
    labels_df_bal['label'] = labels
    labels_df_bal.index  = ids['id'].astype(str) # important

    #print(labels_df_bal.label.value_counts())
    df_all = merge_dfs(labels_df_bal, feats_df)
    #kfolds = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    return df_all

def merge_dfs(labels, feats):
    # the labels df has no header

    return feats.join(labels, on='id', how='inner')
