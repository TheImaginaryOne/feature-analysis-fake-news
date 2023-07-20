# how to run

Requirements: needs Fakeddit tsv comment data files: https://drive.google.com/drive/folders/1DuH0YaEox08ZwzZDpRMOaFpMCeRyxiEF

0. Download comment sentiments + `multi\_modal\_train.tsv` from Fakeddit download website
Also download images into `public\_image\_set/` folder.
Create a folder in the current directory called `output`.
(Extract `all_comments.tsv` into current directory. Because there are some lone `\r` escape sequences, these must be removed and replaced with `\r\n` into a new file called `cleaned_comments.tsv` in the same directory.)

Run `./download_models.sh` in `visual\_sentiment\_analysis` folder.
1. run getimagelist.py (which creates a file called `output/_images.txt`)
2. ./predict_images.sh to predict sentiments. This invokes the image sentiment analysis neural network in the `visual-sentiment-analysis` folder.
Its outputs will appear in `output/imagesents.txt`
3. `calc\_comment\_data.py`. Sentiments of comments will appear in `output/comment_data.csv`
4. run `merge_all.py` to aggregate all data into plots and csv.
5. classify.py to run classifiers
6. plot_clf_results.py to plot reesults graph and get table
6. plot_feat_impts.py to plot feature improtaances

Note: packages required are: sklearn (preferably 1.0.2), pandas, pytorch(-cpu), tqdm, matplotlib, seaborn
