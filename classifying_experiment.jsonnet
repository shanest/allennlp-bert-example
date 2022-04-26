# Modified from the example at
# allennlp/tests/fixtures/bert/bert_for_classification.jssonnet

local bert_model = "bert-base-uncased";
{
    "dataset_reader": {
        "type": "sst_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        }
    },
    "train_data_path": "sst/trees/train.txt",
    "validation_data_path": "sst/trees/dev.txt",
    "model": {
        "type": "bert_classifier",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "freeze_encoder": true,
    },
    "data_loader": {
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
        "num_epochs": 30,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
