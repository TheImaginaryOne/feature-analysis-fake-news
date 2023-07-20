import os, random, sys, argparse
import pathlib

from tqdm import tqdm

from multiprocessing import Process
from multiprocessing import Pool, cpu_count

models_dir = '../visual_sentiment_analysis/'
# add to Python PATH
sys.path.append(models_dir)
# from the above directory
from predict import calc_image_sentiment


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dataset', type=str)

args = parser.parse_args()

# base dir of dataset
base_dir = args.dataset + '/images/'
output_dir = 'output'

class Object(object):
    pass

def run():
    file_names = os.listdir(base_dir)
    
    # write the list of files
    with open(output_dir + "/" + args.dataset + "_image_list.txt", "w") as f:
        f.write("filename\n")
        for file_name in file_names:
            f.write(file_name + "\n")

    cnn_config = Object()
    cnn_config.image_list = file_names
    cnn_config.root = base_dir
    cnn_config.model = 'vgg19_finetuned_all'
    cnn_config.batch_size = 48

    f = open(output_dir + "/" + args.dataset + "_image_sents.csv", "w")
    f.write("image_neg,image_neu,image_pos\n")

    calc_image_sentiment(cnn_config, f, models_dir)

    

if __name__ == '__main__':
    run()
