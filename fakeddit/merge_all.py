import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sns.set_theme()


def calc_sent(df):
    ''' Title sentiment'''
    analyzer = SentimentIntensityAnalyzer()
    agg_sents = df['title'].map(lambda x: analyzer.polarity_scores(x))

    df['title_comp'] = agg_sents.map(lambda x: x['compound'])
    df['title_neg'] = agg_sents.map(lambda x: x['neg'])
    df['title_neu'] = agg_sents.map(lambda x: x['neu'])
    df['title_pos'] = agg_sents.map(lambda x: x['pos'])
    return df

def read_sentiments():
    print("reading stuff...")

    # multiple dfs; concat vertical
    # sents_dfs = [pd.read_csv(filename, header=None) for filename in ["imagesents_0.csv", "imagesents_1.csv"]]
    img_sentiments = pd.read_csv("output/imagesents.txt", header=None)
    img_sentiments.columns = ['image_neg', 'image_neu', 'image_pos']
    #print(img_sentiments[])

    image_list = pd.read_csv("output/_images_list.txt", header=None, names=['image_path'])
    image_list['id'] = image_list['image_path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
    #print(image_list)

    # concat sentiments
    img_sentiments_combined = pd.concat([img_sentiments, image_list], axis=1)
    #print(img_sentiments_combined[img_sentiments_combined['id'].duplicated(keep=False)])
    print(img_sentiments_combined)
    img_sentiments_combined = img_sentiments_combined.set_index('id')

    #some images appear with the same id but with different extensions; remove the duplicates
    img_sentiments_combined = img_sentiments_combined[~img_sentiments_combined.index.duplicated(keep='first')]#[img_sentiments_combined.duplicated()]

    print(img_sentiments_combined[img_sentiments_combined.index.duplicated(keep=False)])
    print(img_sentiments_combined)
    return img_sentiments_combined

img_sentiments_combined = read_sentiments()

print("reading comment datas")
comment_data = pd.read_csv("output/comment_data.csv")
comment_data.rename(columns = {'submission_id': 'id'}, inplace=True)
print("setting idnex of comment datas")
comment_data = comment_data.set_index('id')

print("reading multimodal_train")
# original dataframe
post_data = pd.read_csv("multimodal_train.tsv", sep="\t").set_index('id')
print("post data length:", len(post_data))

print("reading multimodal_test")
# original dataframe
post_data_test = pd.read_csv("multimodal_test_public.tsv", sep="\t").set_index('id')
print("post data test length:", len(post_data_test))

post_data_validate = pd.read_csv("multimodal_validate.tsv", sep="\t").set_index('id')
print("post data val length:", len(post_data_validate))

def merge_dfs(post_data):
    print("img sentiment merge to post data")
    # perform a merge here
    merged = img_sentiments_combined.join(post_data, how='inner')
    print(f"Number of images posts that are included: {merged.index.nunique()}")
    print("post data merge to comment datas")
    merged = merged.join(comment_data, how='left')
    merged = calc_sent(merged)
    return merged

def filter_df(df):
    df = df[~df['subreddit'].isin(['psbattle_artwork', 'photoshopbattles'])]
    return df

merged = filter_df(merge_dfs(post_data))
merged_test = filter_df(merge_dfs(post_data_test))
merged_validate = filter_df(merge_dfs(post_data_validate))

print("merged data length:", len(merged))
print("merged test data length:", len(merged_test))
print("merged val data length:", len(merged_validate))
merged.to_csv('output/merged.csv')
merged_test.to_csv('output/merged_test.csv')
merged_validate.to_csv('output/merged_validate.csv')

# do this in case
merged = merged[~merged.index.duplicated()]

#print(merged)
# ===
#print(merged[merged.index.duplicated()])
#print(merged.columns)
