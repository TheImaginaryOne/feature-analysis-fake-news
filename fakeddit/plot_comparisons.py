import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

a = ['image_neg','image_neu','image_pos']
b = ['title_neg','title_neu','title_pos','title_comp']#,'avg_comp_comment','avg_neg_comment','avg_neu_comment','avg_pos_comment','max_comp_comment','max_neg_comment','max_neu_comment','max_pos_comment','min_comp_comment','min_neg_comment','min_neu_comment','min_pos_comment']
c = ['upvote_ratio','score','num_comments']
d = ['domain_proportion_true', 'author_proportion_true']
feats = a + b + c + d

def get_proportion_encoding(df, col, target):
    ''' Assume the target is binary, ie. 0 / 1 '''
    summary = df.groupby([col]).agg(count=(col, 'count'), amount=(target, 'sum'))
    proportion_col = col + '_proportion_true'# + target
    summary[proportion_col] = summary['amount'] / summary['count']
    return summary[[proportion_col]]

def plot(merged):
    print("plotting....")

    box_cols = feats
    plot_df = pd.melt(merged[box_cols + ["2_way_label"]], id_vars=['2_way_label'])
    #print(plot_df)
    g = sns.catplot(data=plot_df, col='variable', sharey=False, col_wrap=5, x='2_way_label', y='value', kind='box', height=3, legend=True)

    plt.savefig('output/corr_with_label_plot.png', bbox_inches='tight')
    
    # pair plots
    
    #sns.pairplot(data=plot_df, kind='hist')
    
    #plt.savefig('output/sentiment_pair_plot.png')
    
    plt.clf() # clear figure
    
    # corr plots
    corr = merged[feats].corr()
    fig, ax = plt.subplots(1, figsize=(15, 10))
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, fmt=".4f", ax=ax)
    
    plt.savefig('output/sentiment_corr.png', bbox_inches='tight')

# todo read every single file
df_train = pd.read_csv("output/merged.csv")
df_val = pd.read_csv("output/merged_validate.csv")
df_test = pd.read_csv("output/merged_test.csv")

for df_ in [df_train, df_val, df_test]:
    df_.set_index(['id'], inplace=True)
# tododf_train, df_val, 
df = pd.concat([df_train, df_val,df_test], axis='rows')
assert len(df) == len(df_train) + len(df_val) + len(df_test), \
        f"{len(df)} is not equal {len(df_train)} + {len(df_val)} + {len(df_test)}"

# exclude these subs from the analysis
#df = df[~df['subreddit'].isin(['psbattle_artwork', 'photoshopbattles'])]
#df = df[~df['subreddit'].isin(['theonion', 'neutralnews'])]

for col in ['domain', 'author']:
    len_before = len(df)
    #print(df)
    df = df.join(get_proportion_encoding(df, col, '2_way_label'), on=col, how='left')
    len_after = len(df)
    assert len_before == len_after, f'{len_before} != {len_after}'

df['2_way_label'] = df['2_way_label'].map(lambda x: 'real' if x == 1 else 'fake')
#df = df[~df.isna()] # the author label is nan in some posts.
print(df.isna().sum())
plot(df)
