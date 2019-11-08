# allennlp-bert-example
 
Quick usage:
1. Install `allennlp` (0.9.1) in your local virtual/conda environment
    * Or `pip install -r requirements.txt`
2. Run `python get_data.py` to download the semantic tagging dataset and split it into train/val/test
3.
    1. `python run.py` (only tested with 3.7)
    2. `allennlp train experiment.jsonnet -s /tmp/exp --include-package probing`
