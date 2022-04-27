local model_name = "bert-base-cased";
{
    "dataset_reader": {
        "type": "semantic_tag",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": model_name,
            }
        },
    },
    "train_data_path": "sem-0.1.0/data/gold/train",
    "validation_data_path": "sem-0.1.0/data/gold/val",
    "test_data_path": "sem-0.1.0/data/gold/test",
    "model": {
        "type": "subword_word_tagger",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": model_name
                }
            }
        },
        "freeze_encoder": true,
    },
    "data_loader": {
        "batch_size": 32
    },
    "trainer": {
        "num_epochs": 50,
        "patience": 3,
        "optimizer": {
            "type": "adam"
        }
    }
}
