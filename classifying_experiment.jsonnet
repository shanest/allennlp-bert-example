/** You could basically use this config to train your own BERT classifier,
    with the following changes:

    1. change the `bert_model` variable to "bert-base-uncased" (or whichever you prefer)
    2. swap in your own DatasetReader. It should generate instances
       that look like {"tokens": TextField(...), "label": LabelField(...)}.

       You don't need to add the "[CLS]" or "[SEP]" tokens to your instances,
       that's handled automatically by the token indexer.
    3. replace train_data_path and validation_data_path with real paths
    4. any other parameters you want to change (e.g. dropout)
 */


# For a real model you'd want to use "bert-base-uncased" or similar.
local bert_model = "bert-base-uncased";
local do_lowercase = true;
{
    "dataset_reader": {
        "type": "sst_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "do_lowercase": do_lowercase
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
                "do_lowercase": do_lowercase
            }
        }
    },
    "train_data_path": "sst/trees/train.txt",
    "validation_data_path": "sst/trees/dev.txt",
    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "trainable": false,
        "dropout": 0.0
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
        "num_epochs": 30,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
