import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Test the classifier.')
parser.add_argument('dataset', type=str)
args = parser.parse_args()
dataset = args.dataset


df = pd.read_csv(f'output/{dataset}_labels.csv')

print(df['label'].value_counts())

