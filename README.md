# allennlp-bert-example
 
Quick usage:
1. Install `allennlp` (2.9.3) in a fresh virtual/conda environment (NB: we are testing with Python 3.9.x; there appears to be an installation issue with 3.10.x)
    * Or `pip install -r requirements.txt`
2. For a sentiment analysis classifying experiment:
    1. Run `sh get_classifying_data.sh` to download the semantic tagging dataset and split it into train/val/test
    2. Two options for running:
        1. `python run_classifying.py`
        2. `allennlp train classifying_experiment.jsonnet -s /tmp/exp --include-package classifying`
3. For a semantic tagging experiment:
    1. Run `python get_tagging_data.py` to download the semantic tagging dataset and split it into train/val/test
    2. Two options for running:
        1. `python run_tagging.py`
        2. `allennlp train tagging_experiment.jsonnet -s /tmp/exp --include-package tagging`
4. To look at logs during training: `tensorboard --logdir /tmp/exp` (you can run tensorboard on a remote server and use `ssh` port forwarding to view the logs locally)

NB: the first time you run the experiments, the pre-trained models will be downloaded, but retrieved from local cache thereafter.
