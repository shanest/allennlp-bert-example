# allennlp-bert-example
 
Quick usage:
1. Install `allennlp` (0.9.1) in your local virtual/conda environment
    * Or `conda create --name <env_name> --file environment.txt`
2. Run ./get_data.sh to download the semantic tagging dataset
3.
    1. `python run.py` (only tested with 3.7)
    2. `allennlp train experiment.jsonnet -s /tmp/exp --include-package probing`
