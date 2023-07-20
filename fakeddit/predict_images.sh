#!/bin/sh
cd visual-sentiment-analysis && python3 predict.py ../output/_images_list.txt --model vgg19_finetuned_all --batch-size 64 > ../output/imagesents.txt 
#cd visual-sentiment-analysis && python3 predict.py ../_images.txt --model vgg19_finetuned_all --batch-size 64 -i 1 -t 2 > ../imagesents_new_1.txt
