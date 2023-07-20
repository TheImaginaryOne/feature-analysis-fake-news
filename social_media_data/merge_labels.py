import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Test the classifier.')
parser.add_argument('dataset', type=str)
args = parser.parse_args()
dataset = args.dataset

splits_dir = f'{dataset}/splits'

def get_labels():
    df1 = pd.read_csv(splits_dir + '/train_0.txt', header=None)
    df2 = pd.read_csv(splits_dir + '/test_0.txt', header=None)
    df3 = pd.read_csv(splits_dir + '/val.txt', header=None)

    df = pd.concat([df1, df2, df3], axis=0)
    df.columns = ['id', 'label']
    df['id'] = df.id.astype(str)
    df.set_index('id', inplace=True)
    return df

df = get_labels()

df.to_csv(f'output/{dataset}_labels.csv')



