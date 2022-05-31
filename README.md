# text_classification

Project aims to collect a literature contained arthropod species and gene name. We use the [BERT](https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f) to perform text classification.

### Preprocess

Use `preprocess.py` to preprocess dataset to input format and split to 5 folds for cross validation.

### Use BERT for text classification

In `main.py`, you can modify the parameter of model, such as training epoch and learning rate.

### Run on Atlas

Create environment by Anaconda, and install packages of `package_pip.txt`.
Use the command `sbatch run_tc.sh`
