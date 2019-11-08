{
    "dataset_reader": {
        "type": "semantic_tag",
        "model_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "bert-base-cased",
            "start_tokens": [],
            // TODO: do_lowercase shouldn't be here, but from_params complains if it's not
            "do_lowercase": false,
            "end_tokens": [] 
        },
        "model_token_indexers": {
            "model_tokens": {
                "type": "pretrained_transformer",
                "model_name": "bert-base-cased",
                "do_lowercase": false 
            }
        },
        "start_tokens": ["[CLS]"],
        "end_tokens": ["[SEP]"]
    },
    "train_data_path": "sem-0.1.0/data/gold",
    "validation_data_path": "sem-0.1.0/data/gold",
    "model": {
        "type": "subword_word_tagger",
        "subword_embeddings": {
            "type": "basic",
            "model_tokens": {
                "type": "pretrained_transformer",
                // TODO: link this to tokenizer
                "model_name": "bert-base-cased"
            }
        },
        "subword_aggregator": {
            "type": "endpoint",
            // TODO: is there a way to extract this automatically?
            "input_dim": 768,
            "combination": "x"
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8
    },
    "trainer": {
        "num_epochs": 50,
        "patience": 3,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adam"
        }
    }
}
