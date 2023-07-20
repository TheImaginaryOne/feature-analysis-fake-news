from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import pandas as pd
#import dask.dataframe as dd
from tqdm import tqdm

tqdm.pandas()
#from tqdm.dask import TqdmCallback

#cb = TqdmCallback(desc="global")
#cb.register()

analyzer = SentimentIntensityAnalyzer()
def yy(x):
    #print(x)
    score = analyzer.polarity_scores(x)
    return score

def calc_sent(df):
    agg_sents = df['body'].progress_map(yy)
    #ddata = dd.from_pandas(df['body'], npartitions=8)
    #agg_sents = ddata.map(yy).compute(scheduler='processes')  


    df['text_comp'] = agg_sents.map(lambda x: x['compound'])
    df['text_neg'] = agg_sents.map(lambda x: x['neg'])
    df['text_neu'] = agg_sents.map(lambda x: x['neu'])
    df['text_pos'] = agg_sents.map(lambda x: x['pos'])
    return df

# pbpython.com
def wavg(group, avg_name, weight_name):
        d = group[avg_name]
        w = group[weight_name]
        try:
            return (d * w).sum() / w.sum()
        except ZeroDivisionError:
            return d.mean()

def calc_comment_summary(comments):
    comments_sent = calc_sent(comments)
    print("got comment sent")
    print(comments_sent)

    grp_by = comments_sent.groupby('submission_id')
    sentiment_agg = grp_by.agg(
            avg_comp_comment=('text_comp', 'mean'),
            avg_neg_comment=('text_neg', 'mean'),
            avg_neu_comment=('text_neu', 'mean'),
            avg_pos_comment=('text_pos', 'mean'),

            min_comp_comment=('text_comp', 'min'),
            min_neg_comment=('text_neg', 'min'),
            min_neu_comment=('text_neu', 'min'),
            min_pos_comment=('text_pos', 'min'),

            max_comp_comment=('text_comp', 'max'),
            max_neg_comment=('text_neg', 'max'),
            max_neu_comment=('text_neu', 'max'),
            max_pos_comment=('text_pos', 'max'),
            ups_min=('ups', 'min'),
            ups_avg=('ups', 'mean'),
            ups_max=('ups', 'max'))

#    for x in ['neg', 'neu', 'pos']:
#        sentiment_agg[f'avg_{x}_comment'] = grp_by.apply(wavg, f'text_{x}', 'ups')

    return sentiment_agg


if __name__ == "__main__":
    image_list = pd.read_csv("output/_images_list.txt", header=None, names=['image_path'])
    # filter by the list of ids (?! submission_id or id?)
    image_list['id'] = image_list['image_path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
    # ??
    submission_ids = image_list[['id']]
    submission_ids.columns = ['submission_id']
    submission_ids = submission_ids.set_index('submission_id')
    print(submission_ids)

    comments = pd.read_csv("cleaned_comments.tsv", sep='\t')
    comments.body = comments.body.fillna("")

    filtered = comments.set_index('submission_id').join(submission_ids, how='inner')
    print("joined")

    agg_Sents = calc_comment_summary(filtered)
    agg_Sents.to_csv('output/comment_data.csv')
