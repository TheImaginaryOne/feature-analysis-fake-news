import os, random, sys, argparse
import pathlib
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dataset', type=str)

args = parser.parse_args()

# base dir of dataset
dataset = args.dataset
base_dir = args.dataset + '/images/'
output_dir = 'output'

def run():
    def combine_image_sents():
        img_list = pd.read_csv(f"{output_dir}/{dataset}_image_list.txt")
        # rename
        img_list.columns = ['id']
        img_list['id'] = img_list['id'].astype(str)
        # strip file extension
        img_list['id'] = img_list['id'].map(lambda x: x.split(".")[0])
        
        img_sents = pd.read_csv(f"{output_dir}/{dataset}_image_sents.csv")

        return pd.concat([img_list, img_sents], axis=1)

    feats = pd.read_csv(f"{output_dir}/{dataset}_feats.csv")
    feats['id'] = feats['id'].astype(str)
    feats.set_index('id', inplace=True)
    #print(feats)

    img_sents = combine_image_sents().set_index('id')
    #print(img_sents)

    all_data = feats.join(img_sents, how='inner', on='id')
    #print(all_data)

    all_data.to_csv(f"{output_dir}/{dataset}_feats_all.csv")
    

if __name__ == '__main__':
    run()
