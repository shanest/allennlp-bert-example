local model_name = "bert-base-cased";
local do_lowercase = false;
{
    "dataset_reader": {
        "type": "semantic_tag",
        "model_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "start_tokens": [],
            // TODO: do_lowercase shouldn't be here, but from_params complains if it's not
            "do_lowercase": do_lowercase,
            "end_tokens": [] 
        },
        "model_token_indexers": {
            "model_tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "do_lowercase": do_lowercase 
            }
        },
        "start_tokens": ["[CLS]"],
        "end_tokens": ["[SEP]"]
    },
    "train_data_path": "sem-0.1.0/data/gold/train",
    "validation_data_path": "sem-0.1.0/data/gold/val",
    "test_data_path": "sem-0.1.0/data/gold/test",
    "model": {
        "type": "subword_word_tagger",
        "subword_embeddings": {
            "type": "basic",
            "model_tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name
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
