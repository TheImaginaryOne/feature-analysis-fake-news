import json
import pandas as pd
import argparse

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dataset', type=str)

args = parser.parse_args()

# ===
class SentiAnalysis:
    def __init__(self, prefix):
        self.analyser = SentimentIntensityAnalyzer()
        self.prefix = prefix

    def calc_sent(self, in_df):
        agg_sents = in_df.map(lambda x: self.analyser.polarity_scores(x))
        df = pd.DataFrame()
        df[self.prefix + 'neg'] = agg_sents.map(lambda x: x['neg'])
        df[self.prefix + 'neu'] = agg_sents.map(lambda x: x['neu'])
        df[self.prefix + 'pos'] = agg_sents.map(lambda x: x['pos'])
        df[self.prefix + 'comp'] = agg_sents.map(lambda x: x['compound'])
        return df

def read_json(f):
    # todo
    content = f.read()
    object = json.loads(content)
    
    datas = []

    for twt_id, obj in object.items():
        data = {}
        # print(twt_id)
        for key in ["full_text", "retweet_count", "favorite_count"]:
            data[key] = obj[key]
        for key in ["followers_count", "friends_count", "listed_count", "favourites_count", "statuses_count", "verified"]:
            data["user_" + key] = obj["user"][key]

        data["user_id"] = obj["user"]["id"]

        data["user_create_date"] = obj["user"]["created_at"]
        data['post_date'] = obj["created_at"]
        
        urls = obj["user"]["entities"]["description"]["urls"]
        if "url" in obj["user"]["entities"]:
            urls += obj["user"]["entities"]["url"]["urls"]
        
        if len(urls) > 0:
            data['user_url'] = urls[0]["expanded_url"]
        else:
            data['user_url'] = None
        data['id'] = twt_id
        datas.append(data)

    df = pd.DataFrame(datas)
    return df

dataset = args.dataset
with open(f"{dataset}/data.json") as f:
    df = read_json(f)
    print("calcing sent")
    sents = SentiAnalysis("text_").calc_sent(df['full_text'])
    df = pd.concat([sents, df], axis=1)#, ignore_index=True)

    # calculate user age when the post was created
    df["post_date"] = pd.to_datetime(df["post_date"])
    df["user_create_date"] = pd.to_datetime(df["user_create_date"])
    df["user_age_days"] = (df["post_date"] - df["user_create_date"]).dt.days

    print(df)
    df.to_csv(f"output/{dataset}_feats.csv", index=False)
