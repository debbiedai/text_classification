# text_classification

Project aims to collect a literature contained arthropod species and gene name. We use the [BERT](https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f) to perform text classification.

### Preprocess

Use `preprocess.py` to preprocess dataset to input format and split to 5 folds for cross validation.

### Use BERT for text classification

In `main.py`, you can modify the parameter of model, such as training epoch and learning rate.
If you run the model offline, please download BERT pretrained model from HuggingFace first. You can use [bert-base-cased](https://huggingface.co/bert-base-cased) or [bert-base-uncased](https://huggingface.co/bert-base-uncased) depending on your project.

### Run on Atlas

Create environment by Anaconda, and install packages of `package_pip.txt`.
`run_tc.sh` is a [Atlas](https://www.hpc.msstate.edu/computing/atlas/) Job Script, and use the command `sbatch run_tc.sh` to excute it.
