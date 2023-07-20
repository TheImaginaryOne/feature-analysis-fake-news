# Instructions

We use the Mediaeval dataset from `https://zenodo.org/record/4592249` (must send a request to the authors to get this dataset)

* Extract the Mediaeval dataset into a folder called `mediaeval`
* create a folder called `output/`
* process_data.py extracts text and tweet features from the twitter dataset
* predict_img_sents.py calculates image sentiments
* merge_data.py merges the above two files
* classify.py to classify:
`python3 classify.py mediaeval --balance`

* output_clf_res.py to plot results
`python3 output_clf_res.py mediaeval --balance`

requirements: tqdm, pytorch(-cpu), sklearn, pandas, seaborn, vaderSentiment

===

