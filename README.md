# allennlp-bert-example
 
Quick usage:
1. Install `allennlp` (0.9.1) in your local virtual/conda environment
    * Or `pip install -r requirements.txt`
3. For a sentiment analysis classifying experiment:
    1. Run `sh get_classifying_data.sh` to download the semantic tagging dataset and split it into train/val/test
    2. Two options for running:
        1. `python run_classifying.py` (only tested with 3.7)
        2. `allennlp train classifying_experiment.jsonnet -s /tmp/exp --include-package classifying`
2. For a semantic tagging experiment:
    1. Run `python get_tagging_data.py` to download the semantic tagging dataset and split it into train/val/test
    2. Two options for running:
        1. `python run_tagging.py` (only tested with 3.7)
        2. `allennlp train tagging_experiment.jsonnet -s /tmp/exp --include-package tagging`
